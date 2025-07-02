# tests/test_iblt.py (Corrected with verbose printing)

"""
Tests for the Invertible Bloom Lookup Table (IBLT) implementation.
"""

import pytest

from libFilter.core.utils import HashMapping
from libFilter.filters.iblt import IBLT, IBLTItem

@pytest.fixture
def iblt_setup():
    """Provides a standard IBLT and predefined item sets for testing."""
    hash_map = HashMapping.from_seeds(['S1', 'S2', 'S3'], table_size=40)
    
    set_a = {IBLTItem("apple", "red"), IBLTItem(123, 456)}
    set_b = {IBLTItem("grape", "purple"), IBLTItem("apple", "red")}
    set_c_diff_val = {IBLTItem("apple", "green"), IBLTItem(123, 456)}

    return {
        'hash_map': hash_map,
        'set_a': set_a,
        'set_b': set_b,
        'set_c': set_c_diff_val,
    }


def test_peel_destructive(iblt_setup, verbose_printer):
    """Tests that destructive peeling decodes and leaves the IBLT empty."""
    iblt = IBLT(iblt_setup['hash_map'])
    for item in iblt_setup['set_a']:
        iblt.push(item)
    
    verbose_printer(iblt, "IBLT before destructive peel")
    
    added, removed = iblt.peel(destructive=True)
    
    assert added == iblt_setup['set_a']
    assert not removed
    
    verbose_printer(iblt, "IBLT after destructive peel")
    assert all(cell.is_empty() for cell in iblt.cells)


def test_set_difference(iblt_setup, verbose_printer):
    """Tests using IBLT subtraction to find the symmetric difference between sets."""
    map = iblt_setup['hash_map']
    set_a, set_b = iblt_setup['set_a'], iblt_setup['set_b']
    
    iblt_a = IBLT(map)
    for item in set_a:
        iblt_a.push(item)
    verbose_printer(iblt_a, "IBLT A: contains items from Set A")
    
    iblt_b = IBLT(map)
    for item in set_b:
        iblt_b.push(item)
    verbose_printer(iblt_b, "IBLT B: contains items from Set B")
    
    diff_iblt = iblt_a - iblt_b
    verbose_printer(diff_iblt, "Difference IBLT (A - B)")
    
    added, removed = diff_iblt.peel()
    
    assert added == (set_a - set_b)
    assert removed == (set_b - set_a)
    assert IBLTItem(123, 456) in added
    assert IBLTItem("grape", "purple") in removed


def test_value_difference_leads_to_undecodable_cell(iblt_setup, capsys, verbose_printer):
    """
    Tests that items with the same key but different values result in an
    undecodable cell, leading to incomplete peeling.
    """
    map = iblt_setup['hash_map']
    set_a, set_c = iblt_setup['set_a'], iblt_setup['set_c']
    
    iblt_a = IBLT(map)
    for item in set_a:
        iblt_a.push(item)
    verbose_printer(iblt_a, "IBLT A: contains items from Set A")
    
    iblt_c = IBLT(map)
    for item in set_c:
        iblt_c.push(item)
    verbose_printer(iblt_c, "IBLT C: contains items from Set C (value difference)")
    
    diff_iblt = iblt_a - iblt_c
    verbose_printer(diff_iblt, "Difference IBLT (A - C)")
    
    added, removed = diff_iblt.peel()

    assert not added
    assert not removed
    
    captured = capsys.readouterr()
    assert "Warning: IBLT decoding may be incomplete" in captured.out

# ... (other tests like test_copy_method and test_peel_non_destructive do not
# need verbose printing in every step, but the key ones above are now covered) ...

def test_copy_method(iblt_setup):
    """Tests that the copy method creates a distinct but identical IBLT."""
    iblt = IBLT(iblt_setup['hash_map'])
    iblt.push(IBLTItem("key", "val"))
    
    iblt_copy = iblt.copy()
    
    assert id(iblt) != id(iblt_copy)
    assert id(iblt.cells) != id(iblt_copy.cells)
    assert iblt.to_dict() == iblt_copy.to_dict()

def test_peel_non_destructive(iblt_setup, verbose_printer):
    """Tests that non-destructive peeling correctly decodes without altering the original."""
    iblt = IBLT(iblt_setup['hash_map'])
    for item in iblt_setup['set_a']:
        iblt.push(item)
    
    verbose_printer(iblt, "IBLT before non-destructive peel")
    original_state = iblt.to_dict()
    
    added, removed = iblt.peel(destructive=False)
    
    assert added == iblt_setup['set_a']
    assert not removed
    
    verbose_printer(iblt, "IBLT after non-destructive peel (should be unchanged)")
    assert iblt.to_dict() == original_state