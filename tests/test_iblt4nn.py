# tests/test_iblt4nn.py

"""Tests for the IBLT for Neural Networks (IBLT4NN) implementation."""

import pytest
import math
import json

from libFilter.core.prv import PRV
from libFilter.core.utils import HashMapping
from libFilter.filters4nn.iblt4nn import IBLT4NN
from libFilter.filters4nn.nn_utils import NNItem

@pytest.fixture
def nn_setup():
    """Provides a standard setup for IBLT4NN tests."""
    n_indices = 2048
    hash_map = HashMapping.from_seeds(['NNS1', 'NNS2', 'NNS3'], table_size=50)
    prv = PRV(n=n_indices, prp_type='aes128')
    
    client1_updates = [NNItem(idx=10, weight=0.5), NNItem(idx=1024, weight=-0.1)]
    client2_updates = [NNItem(idx=10, weight=0.2), NNItem(idx=100, weight=0.3)]
    
    return {
        'hash_map': hash_map,
        'prv': prv,
        'client1': client1_updates,
        'client2': client2_updates,
    }

def test_aggregation_and_peel(nn_setup, verbose_printer):
    """Tests aggregation of updates and subsequent peeling."""
    map_config, prv = nn_setup['hash_map'], nn_setup['prv']
    client1, client2 = nn_setup['client1'], nn_setup['client2']
    
    agg_iblt = IBLT4NN(map_config, prv)
    
    for item in client1:
        agg_iblt.push(item)
    verbose_printer(agg_iblt, "After pushing Client 1 updates")

    for item in client2:
        agg_iblt.push(item)
    verbose_printer(agg_iblt, "After pushing Client 2 updates (final state)")
    
    decoded = agg_iblt.peel(destructive=False)
    
    # After a non-destructive peel, the original filter should be unchanged.
    verbose_printer(agg_iblt, "Original IBLT state after non-destructive peel")
    
    expected = {10: 0.7, 100: 0.3, 1024: -0.1}
    
    assert len(decoded) == len(expected)
    for idx, weight in expected.items():
        assert idx in decoded
        assert math.isclose(decoded[idx], weight)
        
    # Check if a destructive peel would leave the filter empty
    final_state_checker = agg_iblt.copy()
    final_state_checker.peel(destructive=True)
    verbose_printer(final_state_checker, "State after destructive peel (should be empty)")
    assert final_state_checker.is_fully_decoded()

def test_merge_with_add_operator(nn_setup, verbose_printer):
    """Tests merging two IBLTs using `+` and `+=` operators."""
    map_config, prv = nn_setup['hash_map'], nn_setup['prv']
    client1, client2 = nn_setup['client1'], nn_setup['client2']
    
    iblt1 = IBLT4NN(map_config, prv)
    [iblt1.push(i) for i in client1]
    iblt1_copy = iblt1.copy()
    verbose_printer(iblt1, "IBLT 1 (Client 1) initial state")

    iblt2 = IBLT4NN(map_config, prv)
    [iblt2.push(i) for i in client2]
    verbose_printer(iblt2, "IBLT 2 (Client 2) initial state")
    
    # Test `+` operator
    merged_iblt = iblt1 + iblt2
    verbose_printer(merged_iblt, "Merged IBLT state (from `+`)")
    
    decoded = merged_iblt.peel()
    
    expected = {10: 0.7, 100: 0.3, 1024: -0.1}
    assert len(decoded) == len(expected)
    assert math.isclose(decoded.get(10, 0), expected[10])

    assert iblt1.to_dict() == iblt1_copy.to_dict()
    
    # Test `+=` operator
    iblt1 += iblt2
    verbose_printer(iblt1, "IBLT 1 state after `+=` IBLT 2")
    assert iblt1.to_dict() == merged_iblt.to_dict()

def test_serialization(nn_setup):
    """Tests that the IBLT with PRV can be serialized and deserialized correctly."""
    map_config, prv = nn_setup['hash_map'], nn_setup['prv']
    iblt = IBLT4NN(map_config, prv)
    for item in nn_setup['client1']:
        iblt.push(item)
        
    config = iblt.to_dict()
    rebuilt_iblt = IBLT4NN.from_dict(config)
    
    assert iblt.to_dict() == rebuilt_iblt.to_dict()
    assert rebuilt_iblt.peel() == iblt.peel()

def test_unsupported_operations(nn_setup):
    """Ensures that disabled user-facing operations raise NotImplementedError."""
    iblt = IBLT4NN(nn_setup['hash_map'], nn_setup['prv'])
    
    with pytest.raises(NotImplementedError, match="does not support `remove`"):
        iblt.remove(nn_setup['client1'][0])
    
    with pytest.raises(NotImplementedError, match="does not support the `-` operation"):
        _ = iblt - iblt
        
    with pytest.raises(NotImplementedError, match="does not support the `-=` operation"):
        iblt -= iblt