"""
Demo: RIBLT4NN expand/peel demo for decoding flow.

This script demonstrates:
- riblt1/riblt2 each push items and expand partially
- use slice_to_dict to extract each range (a slice is represented by (start, end) on cells)
- an empty agg riblt absorbs slice data via expand_from_slice
- peel: the first decode cannot fully decode because expand is insufficient
- riblt1/riblt2 expand again, repeat slice and expand_from_slice
- peel: more items can be decoded
- eventually, with enough expand, all items should decode

When debugging failing bench cases, the constants below should match the
corresponding trial in test_riblt4nn_expand_ratio.py:
  n_items=1000 trial=4/8 -> trial_idx=3
  seed=bench_1000_3_335239e27fb4e91e
  expand_step=max(1,n_items//2)=500, max_ratio=4.0, key/PRV/_build_updates match the bench

Run:
    pytest -s tests/bench/test_riblt_demo.py
"""

import math
from .riblt4nn_orig import RIBLT4NN
from .test_riblt4nn_expand_ratio import _build_updates
from libFilter.core.prv import PRV


def test_riblt4nn_expand_and_peel_demo():
    # Align with the FAIL case in test_riblt4nn_expand_ratio: n_items=2000, trial=8/8,
    # trial_idx=7, seed comes from the printed output.
    n_items = 5000
    trial_idx_for_seed = 7  # trial=8/8 (0-based)
    diffusion_seed = "bench_5000_7_9516722b81eb7ea3"
    assert diffusion_seed == f"bench_{n_items}_{trial_idx_for_seed}_9516722b81eb7ea3"

    expand_chunk = max(1, n_items)  # 1000, matches bench expand_step
    max_ratio = 2.0
    max_expand_budget = int(n_items * 2 * max_ratio)  # Matches the bench per-stream expand cap

    n_indices = max(70000, n_items * 20 + 100)
    prf_key = b"a_riblt4nn_key!!"

    client1, client2, verify_weights = _build_updates(n_items)

    # Each side pushes its own data
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

    expanded_round = 0  # Each round expands riblt1/riblt2 by expand_chunk

    # Expand each side
    riblt1.expand(expand_chunk)
    riblt2.expand(expand_chunk)
    expanded_round += expand_chunk
    print(f"riblt1, riblt2 expanded to {len(riblt1.cells)} cells (expanded_round={expanded_round}).")

    # Using slice_to_dict, take cells in [0, current_expand)
    slice1 = riblt1.slice_to_dict(0, len(riblt1.cells))
    slice2 = riblt2.slice_to_dict(0, len(riblt2.cells))

    # agg is a brand-new empty riblt
    agg = RIBLT4NN(prv, diffusion_seed=diffusion_seed)
    agg.expand_from_slice(slice1)
    agg.expand_from_slice(slice2)
    print(f"agg.expand_from_slice: now {len(agg.cells)} cells.")

    # Peel; the first attempt should not fully decode
    decoded = agg.peel()
    decoded_count = len(getattr(agg, "decoded_weights", {}))
    print(
        f"After first peel: decoded_any={decoded}, decoded_count={decoded_count} / {n_items * 2}, "
        f"fully_decoded={agg.is_fully_decoded()}"
    )
    if not agg.is_fully_decoded():
        print("Not fully decoded after first peel.")

    # Expand again
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

        # Append the newly expanded portion into agg (slice only the new part to avoid duplication)
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
            # Print only non-empty cell states (NNSymbol)
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
                    # Print only non-empty undecoded symbol states
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
       
 

        # Validate decoded items
        decoded_keys = set(getattr(agg, "decoded_weights", {}))
        for idx in decoded_keys:
            expect = verify_weights[idx]
            got = agg.decoded_weights[idx]
            assert math.isclose(got, expect, abs_tol=1e-2), f"Idx={idx} got={got} expect={expect}"

        if agg.is_fully_decoded():
            print("All items decoded and values match.")
            break
