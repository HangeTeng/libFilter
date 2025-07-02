# tests/test_cbf.py

"""
Tests for the Counting Bloom Filter (CBF) implementation.
"""

import pytest

from libFilter.core.utils import HashMapping
from libFilter.filters.cbf import CBF, CBFItem

@pytest.fixture
def cbf_setup() -> tuple[CBF, CBFItem, CBFItem]:
    """Provides a standard CBF and some items for testing."""
    hash_map = HashMapping.from_seeds(['S1', 'S2', 'S3'], table_size=100)
    cbf = CBF(hash_map)
    apple = CBFItem("apple")
    banana = CBFItem("banana")
    return cbf, apple, banana


def test_push_and_query(cbf_setup, verbose_printer):
    """Tests adding items and verifying their presence with `in`."""
    cbf, apple, banana = cbf_setup

    cbf.push(apple)
    verbose_printer(cbf, "Push one 'apple'")

    assert apple in cbf
    assert banana not in cbf

def test_removal(cbf_setup, verbose_printer):
    """Tests that removing an item correctly decrements counters."""
    cbf, apple, _ = cbf_setup

    cbf.push(apple)
    verbose_printer(cbf, "After pushing 'apple'")
    assert apple in cbf

    cbf.remove(apple)
    verbose_printer(cbf, "After removing 'apple'")
    assert apple not in cbf
    
    assert all(cell.is_empty() for cell in cbf.cells)

def test_count_estimation(cbf_setup, verbose_printer):
    """Tests the item count estimation functionality."""
    cbf, apple, banana = cbf_setup
    
    cbf.push(apple)
    cbf.push(apple)
    cbf.push(banana)
    verbose_printer(cbf, "Push 2 'apple', 1 'banana'")

    assert cbf.estimate_count(apple) == 2
    assert cbf.estimate_count(banana) == 1

    cbf.remove(apple)
    verbose_printer(cbf, "Remove one 'apple'")
    
    assert cbf.estimate_count(apple) == 1
    assert apple in cbf

    cbf.remove(apple)
    verbose_printer(cbf, "Remove second 'apple'")

    assert cbf.estimate_count(apple) == 0
    assert apple not in cbf

def test_filter_difference(cbf_setup, verbose_printer):
    """Tests the subtraction of two filters to find the set difference."""
    cbf_a, apple, banana = cbf_setup
    
    hash_map_b = cbf_a.hash_mapping.from_config(cbf_a.hash_mapping.get_config())
    cbf_b = CBF(hash_map_b)

    cbf_a.push(apple)
    cbf_a.push(banana)
    verbose_printer(cbf_a, "CBF A: {apple, banana}")
    
    cbf_b.push(apple)
    verbose_printer(cbf_b, "CBF B: {apple}")
    
    diff = cbf_a - cbf_b
    verbose_printer(diff, "Difference (A - B)")
    
    assert apple not in diff
    assert banana in diff
    assert diff.estimate_count(banana) == 1

def test_empty_filter(cbf_setup):
    """Ensures an empty filter behaves correctly."""
    cbf, apple, banana = cbf_setup
    assert apple not in cbf
    assert banana not in cbf
    assert cbf.estimate_count(apple) == 0