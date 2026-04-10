"""
Experiment: how much step-wise expand is needed for successful RIBLT4NN decoding.

Run directly:
  pytest -q libFilter/tests/bench/test_riblt4nn_expand_ratio.py -s

Adjust parameters by editing constants inside the test function.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import statistics
from typing import Dict, List, Optional, Tuple

from libFilter.core.prv import PRV
from .nn_utils_orig import NNItem
from .riblt4nn_orig import RIBLT4NN


def _build_updates(n_items: int) -> Tuple[List[NNItem], List[NNItem], Dict[int, float]]:
    client1 = [NNItem(idx=i * 10, weight=0.5 + (i % 7) * 0.01) for i in range(n_items)]
    client2 = [NNItem(idx=i * 10 + 5, weight=0.2 + (i % 5) * 0.02) for i in range(n_items)]
    verify_weights = {item.idx: item.weight for item in client1 + client2}
    return client1, client2, verify_weights


def _is_decode_correct(riblt: RIBLT4NN, verify_weights: Dict[int, float]) -> bool:
    if not riblt.is_fully_decoded():
        return False
    if set(riblt.decoded_weights.keys()) != set(verify_weights.keys()):
        return False
    tolerance = 1e-6
    for idx, expected in verify_weights.items():
        got = riblt.decoded_weights[idx]
        if not math.isclose(got, expected, abs_tol=tolerance):
            return False
    return True


def _peel_silently(riblt: RIBLT4NN) -> None:
    # The original implementation prints warning messages for incomplete states.
    # Suppress stdout during iterative probing in this benchmark.
    with contextlib.redirect_stdout(io.StringIO()):
        while riblt.peel():
            pass


def _find_expand_single_push_then_expand(
    *,
    n_items: int,
    diffusion_seed: str,
    expand_step: int,
    max_ratio: float,
) -> Optional[int]:
    n_indices = max(70000, n_items * 20 + 100)
    total_items = n_items * 2
    init_max_expand = max(1, int(total_items * max_ratio))
    # Safety guard to avoid infinite experiments on pathological seeds.
    absolute_max_expand = max(init_max_expand, total_items * 200)
    prv = PRV(n=n_indices, prp_type="aes128", key=b"a_riblt4nn_key!!")
    client1, client2, verify_weights = _build_updates(n_items)

    riblt = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    for item in client1:
        riblt.push(item)
    for item in client2:
        riblt.push(item)

    expanded_total = 0
    current_limit = init_max_expand
    while expanded_total < absolute_max_expand:
        target_limit = min(current_limit, absolute_max_expand)
        while expanded_total < target_limit:
            curr_step = min(expand_step, target_limit - expanded_total)
            riblt.expand(curr_step)
            expanded_total += curr_step
            _peel_silently(riblt)
            if _is_decode_correct(riblt, verify_weights):
                return expanded_total
        current_limit = max(current_limit + expand_step, current_limit * 2)
        current_limit = min(current_limit, absolute_max_expand)

    return None


def _find_expand_separate_expand_then_add(
    *,
    n_items: int,
    diffusion_seed: str,
    expand_step: int,
    max_ratio: float,
) -> Optional[int]:
    n_indices = max(70000, n_items * 20 + 100)
    total_items = n_items * 2
    init_max_expand = max(1, int(total_items * max_ratio))
    absolute_max_expand = max(init_max_expand, total_items * 200)
    prv = PRV(n=n_indices, prp_type="aes128", key=b"a_riblt4nn_key!!")
    client1, client2, verify_weights = _build_updates(n_items)

    riblt1 = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    riblt2 = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    for item in client1:
        riblt1.push(item)
    for item in client2:
        riblt2.push(item)

    expanded_total = 0
    current_limit = init_max_expand
    while expanded_total < absolute_max_expand:
        target_limit = min(current_limit, absolute_max_expand)
        while expanded_total < target_limit:
            curr_step = min(expand_step, target_limit - expanded_total)
            riblt1.expand(curr_step)
            riblt2.expand(curr_step)
            expanded_total += curr_step

            riblt_sum = riblt1 + riblt2
            _peel_silently(riblt_sum)
            if _is_decode_correct(riblt_sum, verify_weights):
                return expanded_total
        current_limit = max(current_limit + expand_step, current_limit * 2)
        current_limit = min(current_limit, absolute_max_expand)

    return None


def _p90(values: List[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def test_riblt4nn_decode_expand_ratio_experiment():
    # 参数请直接在这里修改，不要用环境变量
    n_items_list = [50, 100, 200, 500, 1000, 2000, 5000]
    trials = 8
    max_ratio = 4.0
    expand_step = 50
    base_seed = 20260410

    rng = random.Random(base_seed)
    print(
        f"\nRIBLT4NN min-expand experiment: per_client_items={n_items_list}, "
        f"trials={trials}, max_ratio={max_ratio}, expand_step={expand_step}, "
        f"base_seed={base_seed}"
    )

    for n_items in n_items_list:
        expand_single: List[int] = []
        expand_separate: List[int] = []
        ratios_single: List[float] = []
        ratios_separate: List[float] = []
        fail_single = 0
        fail_separate = 0
        for trial_idx in range(trials):
            diffusion_seed = f"bench_{n_items}_{trial_idx}_{rng.getrandbits(64):016x}"
            single_expand = _find_expand_single_push_then_expand(
                n_items=n_items,
                diffusion_seed=diffusion_seed,
                expand_step=expand_step,
                max_ratio=max_ratio,
            )
            separate_expand = _find_expand_separate_expand_then_add(
                n_items=n_items,
                diffusion_seed=diffusion_seed,
                expand_step=expand_step,
                max_ratio=max_ratio,
            )
            if single_expand is None:
                fail_single += 1
                single_str = "FAIL"
            else:
                expand_single.append(single_expand)
                total_items = n_items * 2
                ratio_single = single_expand / total_items
                ratios_single.append(ratio_single)
                single_str = f"{single_expand:>6} ({ratio_single:.4f})"

            if separate_expand is None:
                fail_separate += 1
                separate_str = "FAIL"
            else:
                expand_separate.append(separate_expand)
                total_items = n_items * 2
                ratio_separate = separate_expand / total_items
                ratios_separate.append(ratio_separate)
                separate_str = f"{separate_expand:>6} ({ratio_separate:.4f})"

            print(
                f"  n_items={n_items:>6} trial={trial_idx + 1:>2}/{trials} seed={diffusion_seed} "
                f"single={single_str} separate={separate_str}"
            )

        if ratios_single:
            mean_single = statistics.mean(ratios_single)
            p50_single = statistics.median(ratios_single)
            p90_single = _p90(ratios_single)
            max_single = max(ratios_single)
            single_summary = (
                f"mean={mean_single:.4f} p50={p50_single:.4f} p90={p90_single:.4f} "
                f"max={max_single:.4f} expand_range=[{min(expand_single)}, {max(expand_single)}]"
            )
        else:
            single_summary = "no successful trials"

        if ratios_separate:
            mean_separate = statistics.mean(ratios_separate)
            p50_separate = statistics.median(ratios_separate)
            p90_separate = _p90(ratios_separate)
            max_separate = max(ratios_separate)
            separate_summary = (
                f"mean={mean_separate:.4f} p50={p50_separate:.4f} p90={p90_separate:.4f} "
                f"max={max_separate:.4f} expand_range=[{min(expand_separate)}, {max(expand_separate)}]"
            )
        else:
            separate_summary = "no successful trials"

        print(
            f"[n_items={n_items}] single_push_then_expand: {single_summary}; "
            f"fails={fail_single}/{trials}"
        )
        print(
            f"[n_items={n_items}] separate_expand_then_add: {separate_summary}; "
            f"fails={fail_separate}/{trials}"
        )

