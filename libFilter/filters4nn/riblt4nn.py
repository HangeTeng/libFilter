# libFilter/filters4nn/riblt4nn.py

"""
Implementation of a Rate-less, streaming IBLT for secure numerical aggregation.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Set, List, Tuple

from ..core.prv import PRV
from ..core.base import FilterBase
from ..core.utils import IndexGenerator, Hasher, _normalize_input
from .nn_utils import NNItem, NNSymbol
from ..filters.riblt import SymbolQueue as RIBLTSymbolQueue

class RIBLT4NN(FilterBase[NNSymbol, NNItem]):
    """
    A Rate-less, streaming IBLT for secure numerical aggregation.
    Serialization and copying are only supported for finalized filters.
    """
    __slots__ = (
        'cells', '_symbol_queue', '_prv', '_mask_hasher', '_diffusion_seed',
        '_decoded_weights', '_peeled_indices', 'done_expanding'
    )

    def __init__(self, prv: PRV, mask_seed: Any = "default_mask_seed", diffusion_seed: Any = "default_diffusion_seed"):
        super().__init__()
        self._prv = prv
        self._mask_hasher = Hasher(seed=mask_seed)
        self._diffusion_seed = diffusion_seed
        self._symbol_queue = RIBLTSymbolQueue()
        self.done_expanding = False
        self._decoded_weights: Dict[int, float] = {}
        self._peeled_indices: Set[int] = set()

    # ... (_get_mask_for_idx, _get_generator_for_item are correct) ...
    def _get_mask_for_idx(self, idx: int) -> "galois.FieldArray":
        nonce = 0
        while True:
            data_to_hash = _normalize_input(idx) + b'-' + _normalize_input(nonce)
            r_int = self._mask_hasher.digest_int(data_to_hash, nbytes=16)
            if r_int != 0: return self._prv.GF(r_int)
            nonce += 1
    def _get_generator_for_item(self, item_key: int) -> IndexGenerator:
        return IndexGenerator(key=item_key, seed=self._diffusion_seed)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RIBLT4NN":
        if not data.get('done_expanding', False):
            raise ValueError("Deserialization is only supported for finalized RIBLT4NN instances.")
        prv_config = data['prv']
        key_bytes = prv_config.get('key')
        if isinstance(key_bytes, list): key_bytes = bytes(key_bytes)
        prv = PRV(n=prv_config['n'], prp_type=prv_config['prp_type'], key=key_bytes)
        instance = cls(prv, data['mask_seed'], data['diffusion_seed'])
        GF = prv.GF
        instance.cells = [NNSymbol(GF, **s_data) for s_data in data['cells']]
        instance.set_done_expanding()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the filter's state into a dictionary.
        Only supported for finalized filters.
        """
        if not self.done_expanding:
            raise RuntimeError("Serialization is only supported for finalized RIBLT4NN instances.")

        # --- THIS IS THE FIX ---
        # Changed self.prv to self._prv
        key_list = list(self._prv.key) if self._prv.key else None
        
        return {
            'cells': [s.get_state() for s in self.cells],
            'prv': {'n': self._prv.n, 'prp_type': self._prv.prp_type, 'key': key_list},
            'mask_seed': self._mask_hasher.seed,
            'diffusion_seed': self._diffusion_seed,
            'done_expanding': self.done_expanding,
        }

    def copy(self) -> "RIBLT4NN":
        """
        Creates a deep copy of the filter.
        Only supported for finalized filters.
        """
        if not self.done_expanding:
            raise RuntimeError("Copying is only supported for finalized RIBLT4NN instances.")
        return self.from_dict(self.to_dict())

    # ... (push, expand, set_done_expanding, peel, and other methods are correct) ...
    def push(self, item: NNItem) -> None:
        if self.done_expanding: raise RuntimeError("Cannot push to RIBLT4NN after finalization.")
        r = self._get_mask_for_idx(item.idx)
        source_symbol = NNSymbol.from_item(item, prv=self._prv, r=r)
        generator = self._get_generator_for_item(item.get_key())
        self._symbol_queue.enqueue_and_diffuse(source_symbol, generator, self.cells)
    def expand(self, n: int):
        if self.done_expanding: raise RuntimeError("Cannot expand after finalization.")
        if n <= 0: return
        new_cells = [NNSymbol(self._prv.GF) for _ in range(n)]
        self.cells.extend(new_cells)
        self._symbol_queue.expand_and_diffuse(self.cells)
    def set_done_expanding(self):
        self.done_expanding = True
        self._symbol_queue.clear()
    def peel(self) -> bool:
        items_peeled_this_round = 0
        pure_indices = [
            i for i in range(len(self.cells))
            if i not in self._peeled_indices and self.cells[i].is_pure(self._prv)
        ]
        while pure_indices:
            idx = pure_indices.pop()
            cell = self.cells[idx]
            if not cell.is_pure(self._prv): continue
            symbol_to_peel = cell.copy()
            item = NNSymbol.to_item(symbol_to_peel, self._prv)
            if item.idx in self._decoded_weights: continue
            items_peeled_this_round += 1
            self._decoded_weights[item.idx] = item.weight
            peel_generator = self._get_generator_for_item(item.get_key())
            while peel_generator.curr < len(self.cells):
                affected_idx = peel_generator.curr
                peel_generator.jump()
                self.cells[affected_idx] -= symbol_to_peel
                if self.cells[affected_idx].is_pure(self._prv):
                    pure_indices.append(affected_idx)
        return items_peeled_this_round > 0
    @property
    def decoded_weights(self) -> Dict[int, float]:
        return self._decoded_weights
    def is_fully_decoded(self) -> bool:
        return all(c.is_empty() for c in self.cells)
    def remove(self, item: NNItem):
        raise NotImplementedError("RIBLT4NN is aggregation-only.")
    def __sub__(self, other: FilterBase):
        raise NotImplementedError("RIBLT4NN does not support the `-` operation.")
    def __isub__(self, other: FilterBase):
        raise NotImplementedError("RIBLT4NN does not support the `-=` operation.")