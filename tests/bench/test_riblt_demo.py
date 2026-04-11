"""
Demo: RIBLT4NN expand/peel demo for decoding flow.

此脚本演示：
- riblt1/riblt2 各自push元素，并expand一部分
- 用 slice_to_dict 截取各自的区间（slice不是直接能有的，需要按(start,end)取cells）
- agg 空riblt使用 expand_from_slice 吸收slice数据
- peel：第一次解码因为expand不够，无法全部解码
- riblt1/riblt2 再次expand，重复slice和expand_from_slice
- peel：能够解码更多项
- 最终若全部expand，应该能解码全部

调试 bench 失败用例时，下面常量与 test_riblt4nn_expand_ratio.py 中对应 trial 一致：
  n_items=1000 trial=4/8 -> trial_idx=3
  seed=bench_1000_3_335239e27fb4e91e
  expand_step=max(1,n_items//2)=500, max_ratio=4.0, key/PRV/_build_updates 同 bench

运行方法:
    pytest -s tests/bench/test_riblt_demo.py
"""

import math
from .riblt4nn_orig import RIBLT4NN
from .test_riblt4nn_expand_ratio import _build_updates
from libFilter.core.prv import PRV


def test_riblt4nn_expand_and_peel_demo():
    # 与 test_riblt4nn_expand_ratio 中 FAIL 案例对齐：n_items=2000, trial=8/8, trial_idx=7, seed来自打印
    n_items = 5000
    trial_idx_for_seed = 7  # trial=8/8 (从0开始)
    diffusion_seed = "bench_5000_7_9516722b81eb7ea3"
    assert diffusion_seed == f"bench_{n_items}_{trial_idx_for_seed}_9516722b81eb7ea3"

    expand_chunk = max(1, n_items)  # 1000，与 bench expand_step 一致
    max_ratio = 2.0
    max_expand_budget = int(n_items * 2 * max_ratio)  # 与 bench 单路 expand 上限一致

    n_indices = max(70000, n_items * 20 + 100)
    prf_key = b"a_riblt4nn_key!!"

    client1, client2, verify_weights = _build_updates(n_items)

    # 两个各自 push 数据
    prv = PRV(n=n_indices, prp_type="aes128", key=prf_key)
    riblt1 = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    riblt2 = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    for item in client1:
        riblt1.push(item)
    for item in client2:
        riblt2.push(item)

    print(
        f"Debug config: n_items={n_items} (per client), seed={diffusion_seed}, "
        f"expand_chunk={expand_chunk}, max_expand_budget={max_expand_budget}, "
        f"n_indices={n_indices}"
    )
    print("Initial push done.")

    expanded_round = 0  # 每轮 riblt1/riblt2 各 expand expand_chunk，计一轮

    # 各自expand
    riblt1.expand(expand_chunk)
    riblt2.expand(expand_chunk)
    expanded_round += expand_chunk
    print(f"riblt1, riblt2 expanded to {len(riblt1.cells)} cells (expanded_round={expanded_round}).")

    # 用slice_to_dict方式，取出[0, 当前expand)的cells
    slice1 = riblt1.slice_to_dict(0, len(riblt1.cells))
    slice2 = riblt2.slice_to_dict(0, len(riblt2.cells))

    # agg为全新空riblt
    agg = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    agg.expand_from_slice(slice1)
    agg.expand_from_slice(slice2)
    print(f"agg.expand_from_slice: now {len(agg.cells)} cells.")

    # peel，第一次应该不能全部decode
    decoded = agg.peel()
    decoded_count = len(getattr(agg, "decoded_weights", {}))
    print(
        f"After first peel: decoded_any={decoded}, decoded_count={decoded_count} / {n_items * 2}, "
        f"fully_decoded={agg.is_fully_decoded()}"
    )
    if not agg.is_fully_decoded():
        print("Not fully decoded after first peel.")

    # 再expand一次
    while not agg.is_fully_decoded():
        if expanded_round >= max_expand_budget:
            print(
                f"STOP: expanded_round={expanded_round} >= max_expand_budget={max_expand_budget} "
                f"(matches bench FAIL: still not fully decoded). "
                f"decoded_count={len(getattr(agg, 'decoded_weights', {}))}, "
                f"agg_cells={len(agg.cells)}"
            )
            break

        riblt1.expand(expand_chunk)
        riblt2.expand(expand_chunk)
        expanded_round += expand_chunk
        next_start = len(agg.cells)
        print(
            f"riblt1, riblt2 further expanded to {len(riblt1.cells)} cells "
            f"(expanded_round={expanded_round})."
        )

        # 追加expand部分到agg (只slice新增部分, 避免重复)
        slice1_next = riblt1.slice_to_dict(next_start, len(riblt1.cells))
        slice2_next = riblt2.slice_to_dict(next_start, len(riblt2.cells))
        agg.expand_from_slice(slice1_next)
        agg.expand_from_slice(slice2_next)
        print(f"agg.expand_from_slice: now {len(agg.cells)} cells.")

        decoded_any = agg.peel()
        decoded_count = len(getattr(agg, "decoded_weights", {}))
        print(
            f"Loop peel: decoded_any={decoded_any}, decoded_count={decoded_count} / {n_items * 2}, "
            f"fully_decoded={agg.is_fully_decoded()}"
        )
        if not decoded_any:
            print("DEBUG: This peel did not decode any new items. agg state for debug:")
            from pprint import pprint
            # 只print非空cell状态 (NNSymbol)
            if hasattr(agg, "cells"):
                non_empty_cells = [cell for cell in agg.cells[:10] if not getattr(cell, "is_empty", lambda: False)()]
                if non_empty_cells:
                    print("agg.cells non-empty states (first 10):")
                    states = [getattr(cell, "get_state", lambda: str(cell))() for cell in non_empty_cells]
                    pprint(states)
            if hasattr(agg, "undecoded") and hasattr(agg, "undecoded_indices"):
                if agg.undecoded_indices:
                    print("agg.undecoded count:", len(agg.undecoded_indices))
                    print("agg.undecoded indices sample:", list(agg.undecoded_indices)[:10])
                    # 只打印非空undecoded的symbol状态
                    undecoded_states = []
                    for idx in list(agg.undecoded_indices)[:3]:
                        sym = agg.undecoded.get(idx, None)
                        if sym is not None and hasattr(sym, "is_empty") and not sym.is_empty():
                            if hasattr(sym, "get_state"):
                                undecoded_states.append(sym.get_state())
                    if undecoded_states:
                        print("agg.undecoded (state for first 3 non-empty):")
                        pprint(undecoded_states)
            if hasattr(agg, "decoded_weights"):
                decoded_keys_sample = list(agg.decoded_weights.keys())[:10]
                if decoded_keys_sample:
                    print("agg.decoded_weights keys (sample):", decoded_keys_sample)
       
 

        # 检查已解码项的正确性
        decoded_keys = set(getattr(agg, "decoded_weights", {}))
        for idx in decoded_keys:
            expect = verify_weights[idx]
            got = agg.decoded_weights[idx]
            assert math.isclose(got, expect, abs_tol=1e-2), f"Idx={idx} got={got} expect={expect}"

        if agg.is_fully_decoded():
            print("All items decoded and values match.")
            break
