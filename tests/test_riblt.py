# tests/test_riblt.py

"""Tests for the Robust Invertible Bloom Lookup Table (RIBLT) implementation."""

import pytest
from libFilter.core.utils import IndexGenerator, serialize_typed_value
from libFilter.filters.riblt import RIBLT, RIBLTItem, RIBLTSymbol, _hash_key

@pytest.fixture
def riblt_setup():
    """Provides a standard setup for RIBLT tests."""
    set_a = {RIBLTItem("apple", "red"), RIBLTItem(123, 456), RIBLTItem("common", "item")}
    set_b = {RIBLTItem("grape", "purple"), RIBLTItem(789, "blue"), RIBLTItem("common", "item")}
    set_c_diff_val = {RIBLTItem("apple", "green"), RIBLTItem(123, 456)}
    return {'set_a': set_a, 'set_b': set_b, 'set_c_diff_val': set_c_diff_val}

def test_push_and_expand(verbose_printer):
    """Tests basic push, expand, and the internal state of the symbol queue."""
    riblt = RIBLT(seed="test_push")
    item1 = RIBLTItem("item1", 1)
    riblt.push(item1)
    verbose_printer(riblt, "After pushing item1 with no cells")
    assert len(riblt.cells) == 0
    assert len(riblt._symbol_queue) == 1
    
    riblt.expand(200)
    verbose_printer(riblt, "After expanding to 200 cells")
    assert len(riblt.cells) == 200
    assert any(not c.is_empty() for c in riblt.cells)

def test_set_difference_and_peel(riblt_setup, verbose_printer):
    """Tests using RIBLT for a standard set difference where values match for common keys."""
    set_a, set_b = riblt_setup['set_a'], riblt_setup['set_b']
    riblt = RIBLT(seed="test_diff")
    
    for item in set_a:
        riblt.push(item)
    for item in set_b:
        riblt.remove(item)
    
    riblt.expand(200)
    
    # Peel until no more items can be decoded
    while riblt.peel():
        pass
    
    verbose_printer(riblt, "After peeling standard diff")
    
    expected_added = set_a - set_b
    expected_removed = set_b - set_a
    
    assert riblt.is_fully_decoded(), "RIBLT should be fully decoded for standard difference."
    assert riblt.added_items == expected_added
    assert riblt.removed_items == expected_removed

def test_value_difference_leads_to_incomplete_peel(riblt_setup, verbose_printer):
    """
    Tests that items with the same key but different values result in an
    unresolvable state, leading to incomplete peeling.
    """
    print("\n--- [Test: Value Difference Leads to Incomplete Peel] ---")
    
    set_a = {RIBLTItem("apple", "red"), RIBLTItem("common", "item")}
    set_c_diff_val = {RIBLTItem("apple", "green"), RIBLTItem("uncommon", "item")}
    
    riblt = RIBLT(seed="test_val_diff")

    for item in set_a:
        riblt.push(item)
    for item in set_c_diff_val:
        riblt.remove(item)
        
    riblt.expand(100)

    while riblt.peel():
        pass

    verbose_printer(riblt, "State after peeling with conflict")
    
    # EXPECTATIONS:
    # 1. The filter should NOT be fully decoded due to the "apple" key conflict.
    assert not riblt.is_fully_decoded(), "RIBLT should NOT be fully decoded due to the value conflict."
    
    # 2. The non-conflicting items should be decoded correctly.
    #    - "common" is only in set_a -> should be in added_items.
    #    - "uncommon" is only in set_c_diff_val -> should be in removed_items.
    expected_added = {RIBLTItem("common", "item")}
    expected_removed = {RIBLTItem("uncommon", "item")}
    
    assert riblt.added_items == expected_added
    assert riblt.removed_items == expected_removed

def test_peel_scans_from_end(verbose_printer):
    """
    Verifies that the peel process finds a pure cell at the end of the
    table first.
    """
    riblt = RIBLT()

    key_start_raw, val_start_raw = "key_start", "val_start"
    key_end_raw, val_end_raw = "key_end", "val_end"

    key_start_bytes = serialize_typed_value(key_start_raw)
    val_start_bytes = serialize_typed_value(val_start_raw)
    key_end_bytes = serialize_typed_value(key_end_raw)
    val_end_bytes = serialize_typed_value(val_end_raw)
    
    pure_symbol_start = RIBLTSymbol(1, key_start_bytes, val_start_bytes, _hash_key(key_start_bytes))
    pure_symbol_end = RIBLTSymbol(1, key_end_bytes, val_end_bytes, _hash_key(key_end_bytes))
    
    riblt.expand(100)
    riblt.cells[5] = pure_symbol_start
    riblt.cells[95] = pure_symbol_end
    
    verbose_printer(riblt, "RIBLT with pure cells at start and end")
    
    riblt.peel()
    
    end_item = RIBLTItem(key_end_raw, val_end_raw)
    assert end_item in riblt.added_items, "The item from the end of the table should have been peeled."

def test_push_removes_peeled_item(verbose_printer):
    """Tests that pushing an item already in `removed_items` cancels it out."""
    riblt = RIBLT()
    item_to_remove = RIBLTItem("cancel_me", 1)
    
    riblt._removed_items.add(item_to_remove)
    verbose_printer(riblt, "State with manually peeled `removed` item")
    
    riblt.push(item_to_remove)
    verbose_printer(riblt, "State after pushing the same item")
    
    assert not riblt.removed_items
    assert riblt.is_fully_decoded()