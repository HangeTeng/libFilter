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
    n_indices = 2048
    prv = PRV(n=n_indices, prp_type='aes128', key=b'a_riblt4nn_key!!')
    
    client1_updates = [NNItem(idx=20, weight=0.8), NNItem(idx=1500, weight=-0.3)]
    client2_updates = [NNItem(idx=20, weight=0.1), NNItem(idx=200, weight=0.5)]
    
    return { 'prv': prv, 'client1': client1_updates, 'client2': client2_updates }

def test_push_expand_and_peel(riblt4nn_setup, verbose_printer):
    """Tests the core workflow: push, expand, and peel."""
    prv = riblt4nn_setup['prv']
    client1, client2 = riblt4nn_setup['client1'], riblt4nn_setup['client2']
    
    riblt = RIBLT4NN(prv, diffusion_seed="test_peel_seed")
    verbose_printer(riblt, "Initial state")

    # Step-by-step push and expand
    for item in client1:
        riblt.push(item)
    verbose_printer(riblt, "After pushing Client 1 (before expansion)")

    riblt.expand(50)
    verbose_printer(riblt, "After expanding to 50 cells")

    for item in client2:
        riblt.push(item)
    verbose_printer(riblt, "After pushing Client 2")

    riblt.expand(100) # Expand by another 100
    verbose_printer(riblt, "After final expansion to 150 cells")
    
    peel_round = 1
    while riblt.peel():
        verbose_printer(riblt, f"State after peel round {peel_round}")
        peel_round += 1

    decoded = riblt.decoded_weights
    expected_weights = {20: 0.9, 200: 0.5, 1500: -0.3}
    
    assert len(decoded) == len(expected_weights)
    for idx, weight in decoded.items():
        assert idx in expected_weights
        assert math.isclose(weight, expected_weights[idx])

    assert riblt.is_fully_decoded()

def test_encode_decode_two_clients(riblt4nn_setup, verbose_printer):
    """
    Encodes client1 and client2 updates into two separate RIBLT4NNs,
    adds them together, finalizes, and decodes to check correctness.
    """
    prv = riblt4nn_setup['prv']
    client1, client2 = riblt4nn_setup['client1'], riblt4nn_setup['client2']

    # Encode client1
    riblt1 = RIBLT4NN(prv, diffusion_seed="encode_decode_seed")
    for item in client1:
        riblt1.push(item)
    riblt1.expand(150)
    verbose_printer(riblt1, "Client 1 encoded filter (finalized)")

    # Encode client2
    riblt2 = RIBLT4NN(prv, diffusion_seed="encode_decode_seed")
    for item in client2:
        riblt2.push(item)
    riblt2.expand(150)
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
    expected_weights = {20: 0.8 + 0.1, 1500: -0.3, 200: 0.5}
    assert len(decoded) == len(expected_weights)
    for idx, weight in decoded.items():
        assert idx in expected_weights
        assert math.isclose(weight, expected_weights[idx])
    assert riblt_sum.is_fully_decoded()

def test_serialization_and_copy_on_finalized_filter(riblt4nn_setup, verbose_printer):
    """Tests that copy/serialization works correctly on a finalized filter."""
    prv = riblt4nn_setup['prv']
    
    # Create and finalize a filter
    riblt = RIBLT4NN(prv, diffusion_seed="test_serialize_seed")
    for item in riblt4nn_setup['client1']:
        riblt.push(item)
    for item in riblt4nn_setup['client2']:
        riblt.push(item)
    riblt.expand(150)
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
    expected_weights = {20: 0.9, 200: 0.5, 1500: -0.3}
    assert len(decoded) == len(expected_weights)
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
        if len(decoded) == 3:
            break
    # Should not be able to decode all items
    assert len(decoded) < 3

    # Now expand both riblt1 and riblt2 by 200 more cell
    riblt1.expand(200)
    riblt2.expand(200)

    # Take the last 200 cells as slices
    start, end = 1, 201
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
    expected_weights = {20: 0.9, 200: 0.5, 1500: -0.3}
    assert len(decoded) == len(expected_weights)
    for idx, weight in decoded.items():
        assert idx in expected_weights
        assert math.isclose(weight, expected_weights[idx])
    assert agg_riblt.is_fully_decoded()
