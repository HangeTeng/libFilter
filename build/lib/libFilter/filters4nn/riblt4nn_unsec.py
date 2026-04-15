# libFilter/filters4nn/riblt4nn_unsec.py

"""
Implementation of a Rate-less, streaming IBLT for numerical aggregation (unsecured version).
This version does not require a PRV and uses a count-based symbol (NNSymbol_unsec).
"""

from __future__ import annotations
from typing import Dict, Any, Set, List, Tuple

from ..core.base import FilterBase
from ..core.utils import IndexGenerator
from .nn_utils import NNItem, NNSymbol_unsec
from ..filters.riblt import SymbolQueue as RIBLTSymbolQueue

class RIBLT4NN_unsec(FilterBase[NNSymbol_unsec, NNItem]):
    """
    A Rate-less, streaming IBLT for numerical aggregation (unsecured).
    Serialization and copying are only supported for finalized filters.
    """
    __slots__ = (
        'cells', '_symbol_queue', '_diffusion_seed',
        '_decoded_weights', '_peeled_indices', 'done_expanding'
    )

    def __init__(self, diffusion_seed: Any = "default_diffusion_seed"):
        super().__init__()
        self._diffusion_seed = diffusion_seed
        self._symbol_queue = RIBLTSymbolQueue()
        self.done_expanding = False
        self._decoded_weights: Dict[int, float] = {}
        self._peeled_indices: Set[int] = set()
        self.cells: List[NNSymbol_unsec] = []

    def _get_generator_for_item(self, item_key: int) -> IndexGenerator:
        return IndexGenerator(key=item_key, seed=self._diffusion_seed)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RIBLT4NN_unsec":
        if not data.get('done_expanding', False):
            raise ValueError("Deserialization is only supported for finalized RIBLT4NN_unsec instances.")
        instance = cls(data['diffusion_seed'])
        instance.cells = [NNSymbol_unsec(**s_data) for s_data in data['cells']]
        instance.set_done_expanding()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the filter's state into a dictionary.
        Only supported for finalized filters.
        """
        if not self.done_expanding:
            raise RuntimeError("Serialization is only supported for finalized RIBLT4NN_unsec instances.")
        return {
            'cells': [s.get_state() for s in self.cells],
            'diffusion_seed': self._diffusion_seed,
            'done_expanding': self.done_expanding,
        }

    def copy(self) -> "RIBLT4NN_unsec":
        """
        Creates a deep copy of the filter.
        Only supported for finalized filters.
        """
        if not self.done_expanding:
            raise RuntimeError("Copying is only supported for finalized RIBLT4NN_unsec instances.")
        return self.from_dict(self.to_dict())

    def push(self, item: NNItem) -> None:
        if self.done_expanding:
            raise RuntimeError("Cannot push to RIBLT4NN_unsec after finalization.")
        source_symbol = NNSymbol_unsec.from_item(item)
        generator = self._get_generator_for_item(item.get_key())
        self._symbol_queue.enqueue_and_diffuse(source_symbol, generator, self.cells)

    def expand(self, n: int):
        if self.done_expanding:
            raise RuntimeError("Cannot expand after finalization.")
        if n <= 0:
            return
        new_cells = [NNSymbol_unsec() for _ in range(n)]
        self.cells.extend(new_cells)
        self._symbol_queue.expand_and_diffuse(self.cells)

    def set_done_expanding(self):
        self.done_expanding = True
        self._symbol_queue.clear()

    def peel(self) -> bool:
        items_peeled_this_round = 0
        pure_indices = [
            i for i in range(len(self.cells))
            if i not in self._peeled_indices and self.cells[i].is_pure()
        ]
        while pure_indices:
            idx = pure_indices.pop()
            cell = self.cells[idx]
            if not cell.is_pure():
                continue
            symbol_to_peel = cell.copy()
            item = symbol_to_peel.to_item()
            if item.idx in self._decoded_weights:
                continue
            items_peeled_this_round += 1
            self._decoded_weights[item.idx] = item.weight
            peel_generator = self._get_generator_for_item(item.get_key())
            while peel_generator.curr < len(self.cells):
                affected_idx = peel_generator.curr
                peel_generator.jump()
                self.cells[affected_idx] -= symbol_to_peel
                if self.cells[affected_idx].is_pure():
                    pure_indices.append(affected_idx)
        return items_peeled_this_round > 0

    @property
    def decoded_weights(self) -> Dict[int, float]:
        return self._decoded_weights

    def is_fully_decoded(self) -> bool:
        return all(c.is_empty() for c in self.cells)

    def remove(self, item: NNItem):
        raise NotImplementedError("RIBLT4NN_unsec is aggregation-only.")

    def __sub__(self, other: FilterBase):
        raise NotImplementedError("RIBLT4NN_unsec does not support the `-` operation.")

    def __isub__(self, other: FilterBase):
        raise NotImplementedError("RIBLT4NN_unsec does not support the `-=` operation.")