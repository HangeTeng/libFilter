# tests/test_riblt.py

"""Tests for the Robust Invertible Bloom Lookup Table (RIBLT) implementation."""

import pytest
# Make sure to import IndexGenerator
from libFilter.core.utils import IndexGenerator
from libFilter.filters.riblt import RIBLT, RIBLTItem, RIBLTSymbol

# @pytest.fixture
# def riblt_setup():
#     """Provides a standard setup for RIBLT tests."""
#     set_a = {RIBLTItem("apple", "red"), RIBLTItem(123, 456)}
#     set_b = {RIBLTItem("grape", "purple"), RIBLTItem("apple", "red")}
#     return {'set_a': set_a, 'set_b': set_b}

# def test_push_and_expand(verbose_printer):
#     """Tests basic push, expand, and the internal state of the symbol queue."""
#     riblt = RIBLT()
    
#     # Initially, cells are empty, queue is empty
#     assert len(riblt.cells) == 0
#     assert len(riblt._symbol_queue) == 0
    
#     item1 = RIBLTItem("item1", 1)
#     riblt.push(item1)
#     verbose_printer(riblt, "After pushing item1 with no cells")
    
#     # The item is immediately enqueued because there are no cells.
#     assert len(riblt.cells) == 0
#     assert len(riblt._symbol_queue) == 1
    
#     # Expand the filter. This should process the queued item.
#     riblt.expand(200)
#     verbose_printer(riblt, "After expanding to 200 cells")
    
#     # The queue should still have the item, but its generator has advanced.
#     assert len(riblt.cells) == 200
#     assert len(riblt._symbol_queue) == 1
    
#     # The generator's next index should now be >= 200.
#     assert riblt._symbol_queue.next_coded_index() >= 200
    
#     # Check that some cells are now occupied.
#     assert not all(c.is_empty() for c in riblt.cells)

# def test_set_difference_and_peel(riblt_setup, verbose_printer):
#     """Tests using RIBLT for set difference, similar to IBLT."""
#     set_a, set_b = riblt_setup['set_a'], riblt_setup['set_b']
    
#     riblt = RIBLT()
    
#     # Simulate one side sending its items.
#     for item in set_a:
#         riblt.push(item)
    
#     # Simulate the other side sending its items for removal.
#     for item in set_b:
#         riblt.remove(item)
        
#     verbose_printer(riblt, "After pushing set_a and removing set_b (0 cells)")
    
#     # Expand to allow symbols to be diffused.
#     # A small size might not be enough to decode.
#     riblt.expand(100)
#     riblt.set_done_expanding() # Signal that we are ready to peel.
    
#     verbose_printer(riblt, "After expanding to 100 cells")
    
#     riblt.peel()
#     verbose_printer(riblt, "After peeling")
    
#     # For this simple RIBLT, we check the peeled items.
#     # Note: A true diff requires tracking counts, but we check the content.
#     expected_diff = (set_a - set_b) | (set_b - set_a)
    
#     assert riblt.is_fully_decoded()
#     assert riblt.added_items == expected_diff

# def test_incremental_peel_and_continuation(verbose_printer):
#     """
#     Tests the "continuation" feature: peel, add more data, peel again.
#     """
#     riblt = RIBLT()
    
#     item_a = RIBLTItem("A", 1)
#     item_b = RIBLTItem("B", 2)
#     item_c = RIBLTItem("C", 3)
    
#     # 1. Create a situation that is hard to decode initially.
#     # We create a cycle: A -> B, B -> C, C -> A.
#     # This requires a more complex structure to be resolved.
#     # For simplicity, we just add two items.
#     riblt.push(item_a)
#     riblt.push(item_b)
#     riblt.expand(50) # Small size, likely to have collisions
    
#     verbose_printer(riblt, "Initial state with A and B")
    
#     # 2. Try to peel. It might fail to decode everything.
#     riblt.peel()
    
#     # Check if decoding is incomplete.
#     is_done_first_pass = riblt.is_fully_decoded()
#     peeled_after_first_pass = riblt.added_items.copy()
#     verbose_printer(riblt, f"After first peel. Decoded: {peeled_after_first_pass}")
    
#     # 3. Add more data (the "continuation").
#     # This new item should help resolve the previous ambiguity.
#     print("\n--- [Operation: Continuation - adding more data] ---")
#     riblt.push(item_c)
#     # Also subtract one of the original items to make it more interesting.
#     riblt.remove(item_a)
#     riblt.expand(50) # Add more space
#     riblt.set_done_expanding()

#     verbose_printer(riblt, "After adding C and removing A")
    
#     # 4. Peel again.
#     riblt.peel()
#     verbose_printer(riblt, "After second peel")
    
#     # 5. Verify the final result.
#     # The total set of items is {B, C}.
#     # The set of peeled items should contain B and C.
#     final_peeled = riblt.added_items
    
#     # We expect the final set to contain B and C.
#     # item_a was added and then removed.
#     assert riblt.is_fully_decoded()
#     assert final_peeled == {item_b, item_c, item_a} # `remove` also adds to peeled set


    # New test function to inspect IndexGenerator and diffusion
def test_index_generator_and_diffusion(verbose_printer):
    """
    Tests the IndexGenerator's behavior and verifies that an item is
    correctly diffused into the RIBLT cells at the generated indices.
    """
    print("\n--- [Test: Index Generator and Diffusion] ---")
    
    # 1. Setup
    seed = "test_seed_for_diffusion"
    riblt = RIBLT(seed=seed)
    item = RIBLTItem("hello", "world")
    
    print(f"Created RIBLT with seed: '{seed}'")
    print(f"Test item: {item!r}")
    
    # 2. Manually create the IndexGenerator to predict the indices
    # This mimics the internal logic of the RIBLT's `_diffuse` method.
    symbol = RIBLTSymbol.from_item(item)
    key_for_gen = symbol.key_sum + symbol.value_sum
    index_gen = IndexGenerator(key=key_for_gen, seed=seed)
    
    # 3. Generate and record the first few expected indices
    expected_indices = []
    print("\n--- [Prediction: Generating expected indices] ---")
    for i in range(5):
        idx = index_gen.curr
        expected_indices.append(idx)
        print(f"Expected index {i+1}: {idx}")
        index_gen.jump()
    
    # 4. Push the item to the RIBLT and expand it
    print("\n--- [Action: Pushing item and expanding RIBLT] ---")
    riblt.push(item)
    # Expand to a size large enough to contain all our expected indices
    max_expected_idx = max(expected_indices)
    riblt.expand(max_expected_idx + 10) # Add some buffer
    riblt.set_done_expanding()
    
    verbose_printer(riblt, f"RIBLT state after pushing '{item.key}' and expanding")
    
    # 5. Verification
    print("\n--- [Verification: Checking cell contents at expected indices] ---")
    
    # The symbol that should be present in the cells
    expected_symbol_state = symbol.get_state()
    
    for i, idx in enumerate(expected_indices):
        print(f"Checking cell at predicted index {i+1}: {idx}")
        
        cell_state = riblt.cells[idx].get_state()
        
        # We expect the cell at this index to be identical to the item's symbol
        assert cell_state['count'] == expected_symbol_state['count'], f"Count mismatch at index {idx}"
        assert cell_state['key_sum'] == expected_symbol_state['key_sum'], f"Key sum mismatch at index {idx}"
        assert cell_state['value_sum'] == expected_symbol_state['value_sum'], f"Value sum mismatch at index {idx}"
        
        print(f" -> Success: Cell at index {idx} contains the correct symbol state.")

    # Optional: Verify that a random cell *not* in the sequence is empty
    # Find a non-colliding index for this check
    other_idx = 0
    while other_idx in expected_indices:
        other_idx += 1
    
    print(f"\nChecking a non-affected cell at index: {other_idx}")
    assert riblt.cells[other_idx].is_empty(), f"Cell at index {other_idx} should be empty but is not."
    print(f" -> Success: Cell at index {other_idx} is empty as expected.")