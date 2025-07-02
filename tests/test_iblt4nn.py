# tests/test_iblt4nn.py (With enhanced verbose printing for peel)

"""Tests for the IBLT for Neural Networks (IBLT4NN) implementation."""

import pytest
import math
import json

from libFilter.core.prv import PRV
from libFilter.core.utils import HashMapping
from libFilter.filters4nn.iblt4nn import IBLT4NN, NNItem

@pytest.fixture
def nn_setup():
    """Provides a standard setup for IBLT4NN tests."""
    n_indices = 2048
    hash_map = HashMapping.from_seeds(['NNS1', 'NNS2', 'NNS3'], table_size=50)
    prv = PRV(n=n_indices, prp_type='aes128')
    
    client1_updates = [NNItem(10, 0.5), NNItem(1024, -0.1)]
    client2_updates = [NNItem(10, 0.2), NNItem(100, 0.3)]
    
    return {
        'hash_map': hash_map,
        'prv': prv,
        'client1': client1_updates,
        'client2': client2_updates,
    }

def test_aggregation_and_peel(nn_setup, verbose_printer):
    """Tests aggregation of updates and subsequent peeling with detailed output."""
    map, prv = nn_setup['hash_map'], nn_setup['prv']
    client1, client2 = nn_setup['client1'], nn_setup['client2']
    
    agg_iblt = IBLT4NN(map, prv)
    print("\n--- [Operation: Aggregating client updates] ---")
    print(f"Client 1 updates: {[repr(i) for i in client1]}")
    for item in client1:
        agg_iblt.push(item)
    
    print(f"Client 2 updates: {[repr(i) for i in client2]}")
    for item in client2:
        agg_iblt.push(item)
        
    verbose_printer(agg_iblt, "IBLT state after aggregating all updates")
    
    # --- Enhanced Peel Output ---
    print("\n--- [Operation: Peeling the aggregated IBLT] ---")
    decoded = agg_iblt.peel()
    
    # Use json.dumps for pretty-printing the decoded dictionary
    decoded_str = json.dumps(decoded, indent=2)
    print(f"Decoded items from peel:\n{decoded_str}")
    # --- End of Enhanced Peel Output ---

    expected = {
        10: 0.5 + 0.2, # 0.7
        1024: -0.1,
        100: 0.3
    }
    
    print("\n--- [Operation: Verifying results] ---")
    print(f"Expected items:\n{json.dumps(expected, indent=2)}")
    
    assert len(decoded) == len(expected)
    for idx, val in expected.items():
        assert idx in decoded
        assert math.isclose(decoded[idx], val)

def test_merge_with_add_operator(nn_setup, verbose_printer):
    """Tests merging two IBLTs using `+` and `+=` operators."""
    map, prv = nn_setup['hash_map'], nn_setup['prv']
    client1, client2 = nn_setup['client1'], nn_setup['client2']
    
    iblt1 = IBLT4NN(map, prv); [iblt1.push(i) for i in client1]
    iblt1_copy = iblt1.copy()

    iblt2 = IBLT4NN(map, prv); [iblt2.push(i) for i in client2]
    
    verbose_printer(iblt1, "IBLT 1 state (Client 1)")
    verbose_printer(iblt2, "IBLT 2 state (Client 2)")
    
    # Test `+` operator
    merged_iblt = iblt1 + iblt2
    verbose_printer(merged_iblt, "Merged IBLT state (from `+`)")
    
    # --- Enhanced Peel Output for Merged IBLT ---
    print("\n--- [Operation: Peeling the merged IBLT from `+` operator] ---")
    decoded = merged_iblt.peel()
    decoded_str = json.dumps(decoded, indent=2)
    print(f"Decoded items from merged IBLT:\n{decoded_str}")
    # --- End of Enhanced Peel Output ---
    
    expected = {10: 0.7, 1024: -0.1, 100: 0.3}
    assert len(decoded) == len(expected)
    assert math.isclose(decoded[10], expected[10])

    # Assert non-destructive nature of `+`
    assert iblt1.to_dict() == iblt1_copy.to_dict(), "`+` should be non-destructive."
    
    # Test `+=` operator and verify its result
    iblt1 += iblt2
    assert iblt1.to_dict() == merged_iblt.to_dict(), "`+=` should yield the same result as `+`."


# ... (other tests: test_serialization, test_unsupported_operations remain the same) ...

def test_serialization(nn_setup):
    """Tests that the IBLT with PRV can be serialized and deserialized correctly."""
    map, prv = nn_setup['hash_map'], nn_setup['prv']
    iblt = IBLT4NN(map, prv)
    for item in nn_setup['client1']:
        iblt.push(item)
        
    config = iblt.to_dict()
    rebuilt_iblt = IBLT4NN.from_dict(config)
    
    assert iblt.to_dict() == rebuilt_iblt.to_dict()
    assert rebuilt_iblt.peel() == iblt.peel()

def test_unsupported_operations(nn_setup):
    """Ensures that disabled operations raise NotImplementedError."""
    iblt = IBLT4NN(nn_setup['hash_map'], nn_setup['prv'])
    
    with pytest.raises(NotImplementedError):
        _ = iblt - iblt
    with pytest.raises(NotImplementedError):
        iblt -= iblt
    with pytest.raises(NotImplementedError):
        iblt.remove(nn_setup['client1'][0])