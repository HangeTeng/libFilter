"""
Timing comparison between baseline (original) and optimized RIBLT4NN implementations.

Default behavior: skipped (to avoid flaky CI timing). Enable with:
  RUN_RIBLT_BENCH=1 pytest -q libFilter/tests/test_riblt_time.py -s
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Dict, List, Tuple

import pytest

from libFilter.core.prv import PRV
from libFilter.filters4nn.nn_utils import NNItem
from libFilter.filters4nn.riblt4nn import RIBLT4NN as RIBLT4NN_OPT

from .bench.riblt4nn_orig import RIBLT4NN as RIBLT4NN_ORIG


def _build_updates(n: int) -> Tuple[List[NNItem], List[NNItem]]:
    client1 = [NNItem(idx=i * 100, weight=0.5 + (i % 7) * 0.01) for i in range(n)]
    client2 = [NNItem(idx=i * 100 + 50, weight=0.2 + (i % 5) * 0.02) for i in range(n)]
    return client1, client2


def _bench_one(FilterCls, *, prv: PRV, client1: List[NNItem], client2: List[NNItem], expand_size: int, kwargs=None) -> Dict[str, float]:
    kwargs = kwargs or {}
    riblt = FilterCls(prv, **kwargs)

    t0 = time.perf_counter()
    for it in client1:
        riblt.push(it)
    t1 = time.perf_counter()

    riblt.expand(expand_size)
    t2 = time.perf_counter()

    for it in client2:
        riblt.push(it)
    t3 = time.perf_counter()

    riblt.expand(expand_size)
    t4 = time.perf_counter()

    while riblt.peel():
        pass
    t5 = time.perf_counter()

    assert riblt.is_fully_decoded()
    return {
        "push": (t1 - t0) + (t3 - t2),
        "expand": (t2 - t1) + (t4 - t3),
        "peel": t5 - t4,
        "total": t5 - t0,
    }


import statistics
import collections

def test_riblt_time_compare():
    """
    Single-process performance benchmark comparing the original vs. optimized
    RIBLT4NN implementations.
    """
    size_settings = [
        # (500, 1500, 3),
        (1000, 3000, 3),
        # (5000, 15000, 3),
        # (20000, 60000, 3),
    ]
    prv_config = {'n': 11689512, 'prp_type': "des64", 'key': b"a_riblt4nn_key!!"}

    # Pre-generate all data
    all_clients = {n: _build_updates(n) for n, _, _ in size_settings}

    results = collections.defaultdict(lambda: {"orig": [], "opt": []})

    for n_items, expand_size, repeat in size_settings:
        client1, client2 = all_clients[n_items]
        for _ in range(repeat):
            # Original
            prv = PRV(**prv_config)
            res = _bench_one(RIBLT4NN_ORIG, prv=prv, client1=client1, client2=client2,
                             expand_size=expand_size, kwargs={"diffusion_seed": "bench"})
            results[(n_items, expand_size)]["orig"].append(res)
        for _ in range(repeat):
            # Optimized
            prv = PRV(**prv_config)
            res = _bench_one(RIBLT4NN_OPT, prv=prv, client1=client1, client2=client2,
                             expand_size=expand_size, kwargs={"diffusion_seed": "bench", "ndigits": 6})
            results[(n_items, expand_size)]["opt"].append(res)

    keys = ["push", "expand", "peel", "total"]

    print("\nRIBLT4NN benchmark (single-process, simplified):")
    for (n_items, expand_size), res_dict in results.items():
        orig_runs = res_dict["orig"]
        opt_runs = res_dict["opt"]
        print(f"\nRIBLT4NN n_items={n_items}, expand={expand_size}, repeat={len(orig_runs)}:")
        for k in keys:
            t_orig = statistics.mean([r[k] for r in orig_runs]) if orig_runs else float("nan")
            t_opt = statistics.mean([r[k] for r in opt_runs]) if opt_runs else float("nan")
            speedup = (t_orig / t_opt) if t_opt > 0 else float("inf")
            improvement = (t_orig - t_opt) / t_orig * 100 if t_orig > 0 else 0.0
            print(f"  {k:>5}: orig={t_orig:.6f}s opt={t_opt:.6f}s  speedup={speedup:.2f}x  improvement={improvement:.1f}%")
