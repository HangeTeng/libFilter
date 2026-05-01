# tests/test_riblt4nn.py

"""Tests for the Rate-less IBLT for Neural Networks (RIBLT4NN) implementation."""

import pytest
import math

from libFilter.core.prv import PRV
from libFilter.filters4nn.nn_utils import NNItem
from libFilter.filters4nn.riblt4nn import RIBLT4NN

@pytest.fixture
def riblt4nn_setup():
    """Provides a standard setup for RIBLT4NN tests."""
    is_large_update = False
    # Use a larger n_indices to support large update lists
    if is_large_update:
        n_indices = 11689512
        prv = PRV(n=n_indices, prp_type='aes128', key=b'a_riblt4nn_key!!')
        
        # Generate a large update list for testing
        client1_updates = [NNItem(idx=i*100, weight=0.5 + (i % 7) * 0.01) for i in range(5000)]
        client2_updates = [NNItem(idx=i*100+50, weight=0.2 + (i % 5) * 0.02) for i in range(5000)]
        verify_weights = {item.idx: item.weight for item in client1_updates + client2_updates}
        expand_size = 13500
    else:
        n_indices = 7000
        prv = PRV(n=n_indices, prp_type='aes128', key=b'a_riblt4nn_key!!')
        # 50 items each
        client1_updates = [NNItem(idx=i * 10, weight=0.5 + (i % 7) * 0.01) for i in range(5)]
        client2_updates = [NNItem(idx=i * 10 + 5, weight=0.2 + (i % 5) * 0.02) for i in range(5)]
        verify_weights = {item.idx: item.weight for item in client1_updates + client2_updates}
        expand_size = 30

    return { 'prv': prv, 'client1': client1_updates, 'client2': client2_updates, 'expand_size': expand_size, 'verify_weights': verify_weights}

def test_push_expand_and_peel(riblt4nn_setup, verbose_printer):
    """Tests the core workflow: push, expand, and peel."""
    prv = riblt4nn_setup['prv']
    client1, client2 = riblt4nn_setup['client1'], riblt4nn_setup['client2']
    expand_size = riblt4nn_setup['expand_size']
    verify_weights = riblt4nn_setup['verify_weights']
    riblt = RIBLT4NN(prv, diffusion_seed="test_peel_seed")
    verbose_printer(riblt, "Initial state")

    # Step-by-step push and expand
    for item in client1:
        riblt.push(item)
    verbose_printer(riblt, "After pushing Client 1 (before expansion)")

    riblt.expand(expand_size)
    verbose_printer(riblt, f"After expanding to {expand_size} cells")

    for item in client2:
        riblt.push(item)
    verbose_printer(riblt, "After pushing Client 2")

    riblt.expand(expand_size) # Expand by another expand_size
    verbose_printer(riblt, f"After final expansion to {expand_size} cells")
    
    peel_round = 1
    while riblt.peel():
        verbose_printer(riblt, f"State after peel round {peel_round}")
        peel_round += 1

    decoded = riblt.decoded_weights
    assert set(decoded.keys()) == set(verify_weights.keys())
    tolerance = 1e-6
    for k in decoded:
        assert math.isclose(decoded[k], verify_weights[k], abs_tol=tolerance), f"idx={k}: {decoded[k]} != {verify_weights[k]}"
    assert riblt.is_fully_decoded()

def test_encode_decode_two_clients(riblt4nn_setup, verbose_printer):
    """
    Encodes client1 and client2 updates into two separate RIBLT4NNs,
    adds them together, finalizes, and decodes to check correctness.
    """
    prv = riblt4nn_setup['prv']
    client1, client2 = riblt4nn_setup['client1'], riblt4nn_setup['client2']
    expand_size = riblt4nn_setup['expand_size']
    verify_weights = riblt4nn_setup['verify_weights']
    # Encode client1
    riblt1 = RIBLT4NN(prv, diffusion_seed="encode_decode_seed")
    for item in client1:
        riblt1.push(item)
    riblt1.expand(expand_size)
    verbose_printer(riblt1, "Client 1 encoded filter (finalized)")

    # Encode client2
    riblt2 = RIBLT4NN(prv, diffusion_seed="encode_decode_seed")
    for item in client2:
        riblt2.push(item)
    riblt2.expand(expand_size)
    verbose_printer(riblt2, "Client 2 encoded filter (finalized)")

    # Add the two filters
    riblt_sum = riblt1 + riblt2
    verbose_printer(riblt_sum, "Sum of Client 1 and Client 2 filters (before peel)")

    # Decode (peel)
    peel_round = 1
    while riblt_sum.peel():
        verbose_printer(riblt_sum, f"State after peel round {peel_round}")
        peel_round += 1

    decoded = riblt_sum.decoded_weights
    assert set(decoded.keys()) == set(verify_weights.keys())
    tolerance = 1e-6
    for k in decoded:
        assert math.isclose(decoded[k], verify_weights[k], abs_tol=tolerance), f"idx={k}: {decoded[k]} != {verify_weights[k]}"
    assert riblt_sum.is_fully_decoded()

def test_serialization_and_copy_on_finalized_filter(riblt4nn_setup, verbose_printer):
    """Tests that copy/serialization works correctly on a finalized filter."""
    prv = riblt4nn_setup['prv']
    expand_size = riblt4nn_setup['expand_size']
    verify_weights = riblt4nn_setup['verify_weights']
    # Create and finalize a filter
    riblt = RIBLT4NN(prv, diffusion_seed="test_serialize_seed")
    for item in riblt4nn_setup['client1']:
        riblt.push(item)
    for item in riblt4nn_setup['client2']:
        riblt.push(item)
    riblt.expand(expand_size)
    verbose_printer(riblt, "Finalized filter to be copied/serialized")

    # --- Test Copy ---
    copied_riblt = riblt.copy()
    verbose_printer(copied_riblt, "Copied filter state")
    assert copied_riblt.to_dict() == riblt.to_dict()
    
    # --- Test Serialization ---
    config = riblt.to_dict()
    rebuilt_riblt = RIBLT4NN.from_dict(config)
    verbose_printer(rebuilt_riblt, "Rebuilt filter state from serialization")
    assert rebuilt_riblt.to_dict() == riblt.to_dict()
    
    # --- Verify functionality of the rebuilt filter ---
    peel_round = 1
    while rebuilt_riblt.peel():
        verbose_printer(rebuilt_riblt, f"Rebuilt filter state after peel round {peel_round}")
        peel_round += 1
        
    decoded = rebuilt_riblt.decoded_weights
    assert set(decoded.keys()) == set(verify_weights.keys())
    tolerance = 1e-6
    for k in decoded:
        assert math.isclose(decoded[k], verify_weights[k], abs_tol=tolerance), f"idx={k}: {decoded[k]} != {verify_weights[k]}"
    assert rebuilt_riblt.is_fully_decoded()
        
def test_unsupported_operations(riblt4nn_setup, verbose_printer):
    """Ensures that disabled user-facing operations raise NotImplementedError."""
    riblt = RIBLT4NN(riblt4nn_setup['prv'])
    verbose_printer(riblt, "Testing unsupported operations on this filter")
    
    with pytest.raises(NotImplementedError):
        riblt.remove(riblt4nn_setup['client1'][0])
    with pytest.raises(NotImplementedError):
        _ = riblt - riblt
    with pytest.raises(NotImplementedError):
        riblt -= riblt


def test_expand_from_slice_and_merge_decode(riblt4nn_setup, verbose_printer):
    """
    Test: 
    - Create two RIBLT4NNs, each expand(10), push different items, copy one as c, add the other to c, try to decode (should fail).
    - Then expand(200) on both, take the last 200 cells as slices, expand_from_slice into c, then decoding should succeed.
    """
    prv = riblt4nn_setup['prv']
    client1, client2 = riblt4nn_setup['client1'], riblt4nn_setup['client2']
    expand_size = riblt4nn_setup['expand_size']
    verify_weights = riblt4nn_setup['verify_weights']
    # Create two filters, each with 10 cells
    riblt1 = RIBLT4NN(prv)
    riblt2 = RIBLT4NN(prv)
    riblt1.expand(1)
    riblt2.expand(1)

    # Push client1 to riblt1, client2 to riblt2
    for item in client1:
        riblt1.push(item)
    for item in client2:
        riblt2.push(item)

    # Copy riblt1 to c, add riblt2 to c (cell-wise addition)
    agg_riblt = riblt1.copy()
    # Add riblt2's cells to c's cells
    for i in range(len(agg_riblt.cells)):
        agg_riblt.cells[i] += riblt2.cells[i]

    # Finalize c and try to decode (should fail, not enough cells)
    verbose_printer(agg_riblt, "Merged filter after 10 cells (should not decode)")
    decoded = {}
    for _ in range(3):
        agg_riblt.peel()
        decoded = agg_riblt.decoded_weights
    # Should not be able to decode the full update set yet
    assert len(decoded) < len(verify_weights)

    # Now expand both riblt1 and riblt2 by 200 more cell
    riblt1.expand(expand_size)
    riblt2.expand(expand_size)

    # Take the last 200 cells as slices
    start, end = 1, expand_size + 1
    slice1 = riblt1.slice_to_dict(start, end)
    slice2 = riblt2.slice_to_dict(start, end)

    # Now expand from both slices
    agg_riblt.expand_from_slice(slice1)
    agg_riblt.expand_from_slice(slice2) 
    verbose_printer(agg_riblt, "Merged filter after expanding from slices (should decode)")

    # Now decoding should succeed
    peel_round = 1
    while agg_riblt.peel():
        verbose_printer(agg_riblt, f"State after peel round {peel_round}")
        peel_round += 1
    decoded = agg_riblt.decoded_weights
    # Use a tolerance to compare decoded values with verify_weights
    assert set(decoded.keys()) == set(verify_weights.keys())
    tolerance = 1e-6
    for k in decoded:
        assert math.isclose(decoded[k], verify_weights[k], abs_tol=tolerance), f"idx={k}: {decoded[k]} != {verify_weights[k]}"
    assert agg_riblt.is_fully_decoded()
