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


@pytest.mark.skipif(os.getenv("RUN_RIBLT_BENCH") != "1", reason="Set RUN_RIBLT_BENCH=1 to enable timing bench.")
def test_riblt_time_compare():
    # Match the large-ish fixture pattern but keep it adjustable.
    n_items = int(os.getenv("RIBLT_BENCH_ITEMS", "3000"))
    expand_size = int(os.getenv("RIBLT_BENCH_EXPAND", "15000"))
    repeat = int(os.getenv("RIBLT_BENCH_REPEAT", "1"))

    prv = PRV(n=11689512, prp_type="aes128", key=b"a_riblt4nn_key!!")
    client1, client2 = _build_updates(n_items)

    # Warm-up to reduce first-run effects
    # _bench_one(RIBLT4NN_ORIG, prv=prv, client1=client1, client2=client2, expand_size=expand_size, kwargs={"diffusion_seed": "bench"})
    _bench_one(RIBLT4NN_OPT, prv=prv, client1=client1, client2=client2, expand_size=expand_size, kwargs={"diffusion_seed": "bench", "ndigits": 6})

    orig_runs = [
        _bench_one(RIBLT4NN_ORIG, prv=prv, client1=client1, client2=client2, expand_size=expand_size, kwargs={"diffusion_seed": "bench"})
        for _ in range(repeat)
    ]
    opt_runs = [
        _bench_one(RIBLT4NN_OPT, prv=prv, client1=client1, client2=client2, expand_size=expand_size, kwargs={"diffusion_seed": "bench", "ndigits": 6})
        for _ in range(repeat)
    ]

    def avg(key: str, runs: List[Dict[str, float]]) -> float:
        return statistics.mean(r[key] for r in runs)

    keys = ["push", "expand", "peel", "total"]
    summary = {k: (avg(k, orig_runs), avg(k, opt_runs)) for k in keys}

    print(f"\nRIBLT4NN timing compare (items={n_items*2}, expand={expand_size}, repeat={repeat})")
    for k in keys:
        t_orig, t_opt = summary[k]
        speedup = (t_orig / t_opt) if t_opt > 0 else float("inf")
        improvement = (t_orig - t_opt) / t_orig * 100 if t_orig > 0 else 0.0
        print(f"  {k:>5}: orig={t_orig:.6f}s opt={t_opt:.6f}s  speedup={speedup:.2f}x  improvement={improvement:.1f}%")

