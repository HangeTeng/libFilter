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

    riblt.set_done_expanding()
    verbose_printer(riblt, "After finalizing (set_done_expanding)")
    
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
    riblt.set_done_expanding()
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

def test_serialization_fails_on_streaming_filter(riblt4nn_setup, verbose_printer):
    """
    Ensures that copy() and to_dict() raise errors on non-finalized filters.
    """
    prv = riblt4nn_setup['prv']
    riblt_streaming = RIBLT4NN(prv)
    riblt_streaming.push(riblt4nn_setup['client1'][0])
    verbose_printer(riblt_streaming, "Streaming filter (should fail to serialize)")

    with pytest.raises(RuntimeError, match="Serialization is only supported"):
        riblt_streaming.to_dict()

    with pytest.raises(RuntimeError, match="Copying is only supported"):
        riblt_streaming.copy()
        
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