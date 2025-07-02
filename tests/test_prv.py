# tests/test_prv.py

"""Tests for the Pseudo-Random Vector (PRV) implementation."""

import pytest
import galois
from libFilter.core.prv import PRV

@pytest.fixture(params=list(PRV.CONFIGS.keys()))
def prv_type(request):
    """A fixture to parametrize tests over all supported PRP types."""
    return request.param

def test_determinism_with_default_key(prv_type):
    """Tests that default keys are deterministic for the same PRP type."""
    prv1 = PRV(n=100_000, prp_type=prv_type)
    prv2 = PRV(n=100_000, prp_type=prv_type)
    assert prv1.key == prv2.key

def test_user_provided_key(prv_type):
    """Tests the functionality of using a custom key."""
    key_size = PRV.CONFIGS[prv_type]['key_size']
    
    # Test valid key
    custom_key = b'\xAA' * key_size
    prv = PRV(n=100_000, prp_type=prv_type, key=custom_key)
    assert prv.key == custom_key
    
    # Test invalid key
    invalid_key = b'\xBB' * (key_size + 1)
    with pytest.raises(ValueError, match="requires a .* key"):
        PRV(n=100_000, prp_type=prv_type, key=invalid_key)

def test_forward_and_backward_conversion(prv_type):
    """Tests that entry() and index() are inverse operations."""
    n = 100_000
    prv = PRV(n=n, prp_type=prv_type)
    test_index = 12345
    
    gf_element = prv.entry(test_index)
    found_index = prv.index(gf_element)
    
    assert isinstance(gf_element, galois.FieldArray)
    assert found_index == test_index

def test_galois_field_arithmetic(prv_type):
    """Tests that returned elements support correct field arithmetic."""
    prv = PRV(n=100_000, prp_type=prv_type)
    val1 = prv.entry(9876)
    val2 = prv.entry(5432)
    
    # Check against manually performed operations in the same field
    expected_sum = prv.GF(int(val1)) + prv.GF(int(val2))
    expected_prod = prv.GF(int(val1)) * prv.GF(int(val2))
    
    assert val1 + val2 == expected_sum
    assert val1 * val2 == expected_prod

def test_edge_cases_and_invalid_input(prv_type):
    """Tests boundary conditions and invalid inputs."""
    n = 100_000
    prv = PRV(n=n, prp_type=prv_type)
    
    # Index out of bounds
    with pytest.raises(IndexError):
        prv.entry(n)
    
    # Value out of PRP domain
    invalid_k_value = 1 << prv.prp_bits
    assert prv.index(invalid_k_value) is None
    
    # Non-integer convertible input
    assert prv.index("invalid_string") is None

    # Index maps outside of n
    # This requires finding a k that decodes to i >= n, which is hard.
    # We trust the implementation's `i if i < n else None` logic.