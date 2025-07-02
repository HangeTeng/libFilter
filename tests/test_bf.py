# tests/test_bf.py (Corrected with verbose printing)

"""
Tests for the Bloom Filter (BF) implementation.
"""

import pytest

from libFilter.core.utils import HashMapping
from libFilter.filters.bf import BF, BFItem

@pytest.fixture
def basic_bf() -> BF:
    """A fixture to provide a standard, empty Bloom Filter for testing."""
    hash_map = HashMapping.from_seeds(['S1', 'S2', 'S3'], table_size=100)
    return BF(hash_map)

def test_push_and_membership(basic_bf: BF, verbose_printer):
    """
    Tests adding items to the filter and verifying their presence and
    the absence of other items.
    """
    bf = basic_bf
    items_present = [BFItem("apple"), BFItem(123)]
    items_absent = [BFItem("cherry"), BFItem(456)]

    # 1. Add items
    for item in items_present:
        bf.push(item)
    verbose_printer(bf, "After pushing initial items: 'apple', 123")

    # 2. Check for presence
    for item in items_present:
        assert item in bf, f"Item {item} should be present in the filter."
    
    # 3. Check for absence
    for item in items_absent:
        assert item not in bf, f"Item {item} should be absent from the filter."

def test_removal_is_not_supported(basic_bf: BF):
    """Ensures that the remove operation raises a NotImplementedError."""
    item_to_remove = BFItem("apple")
    
    with pytest.raises(NotImplementedError):
        basic_bf.remove(item_to_remove)

def test_copy_and_merge(basic_bf: BF, verbose_printer):
    """Tests the copy and in-place addition (merge) functionality."""
    bf = basic_bf
    item1 = BFItem("apple")
    item2 = BFItem("banana")

    # 1. Add one item and copy
    bf.push(item1)
    verbose_printer(bf, "Original BF after pushing 'apple'")
    bf_copy = bf.copy()
    verbose_printer(bf_copy, "Copied BF state")
    
    # 2. Add another item to the original
    bf.push(item2)
    verbose_printer(bf, "Original BF after pushing 'banana'")
    
    # 3. Merge original into the copy
    bf_copy += bf
    verbose_printer(bf_copy, "Copied BF after merging with original")
    
    # 4. Verify both items are now in the merged copy
    assert item1 in bf_copy
    assert item2 in bf_copy

def test_empty_filter_contains_nothing(basic_bf: BF):
    """Ensures a new, empty filter does not contain any item."""
    assert BFItem("anything") not in basic_bf