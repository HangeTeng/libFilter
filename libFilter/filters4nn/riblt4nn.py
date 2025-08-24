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
    """
    __slots__ = (
        'cells', '_symbol_queue', '_prv', '_mask_hasher', '_diffusion_seed',
        '_decoded_weights', '_peeled_indices'
    )

    def __init__(self, prv: PRV, mask_seed: Any = "default_mask_seed", diffusion_seed: Any = "default_diffusion_seed"):
        super().__init__()
        self._prv = prv
        self._mask_hasher = Hasher(seed=mask_seed)
        self._diffusion_seed = diffusion_seed
        self._symbol_queue = RIBLTSymbolQueue()
        self._decoded_weights: Dict[int, float] = {}
        self._peeled_indices: Set[int] = set()

    def _get_mask_for_idx(self, idx: int) -> "galois.FieldArray":
        nonce = 0
        while True:
            data_to_hash = _normalize_input(idx) + b'-' + _normalize_input(nonce)
            r_int = self._mask_hasher.digest_int(data_to_hash, nbytes=self._prv.prp_bits//8)
            if r_int != 0: return self._prv.GF(r_int)
            nonce += 1
    def _get_generator_for_item(self, item_key: int) -> IndexGenerator:
        return IndexGenerator(key=item_key, seed=self._diffusion_seed)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RIBLT4NN":
        # No done_expanding check
        prv_config = data['prv']
        key_bytes = prv_config.get('key')
        if isinstance(key_bytes, list): key_bytes = bytes(key_bytes)
        prv = PRV(n=prv_config['n'], prp_type=prv_config['prp_type'], key=key_bytes)
        instance = cls(prv, data['mask_seed'], data['diffusion_seed'])
        GF = prv.GF
        instance.cells = [NNSymbol(GF, **s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the filter's state into a dictionary.
        """
        key_list = list(self._prv.key) if self._prv.key else None
        return {
            'cells': [s.get_state() for s in self.cells],
            'prv': {'n': self._prv.n, 'prp_type': self._prv.prp_type, 'key': key_list},
            'mask_seed': self._mask_hasher.seed,
            'diffusion_seed': self._diffusion_seed,
        }

    def copy(self) -> "RIBLT4NN":
        """
        Creates a deep copy of the filter.
        """
        return self.from_dict(self.to_dict())

    def push(self, item: NNItem) -> None:
        r = self._get_mask_for_idx(item.idx)
        source_symbol = NNSymbol.from_item(item, prv=self._prv, r=r)
        generator = self._get_generator_for_item(item.get_key())
        self._symbol_queue.enqueue_and_diffuse(source_symbol, generator, self.cells)
    def expand(self, n: int):
        if n <= 0: return
        new_cells = [NNSymbol(self._prv.GF) for _ in range(n)]
        self.cells.extend(new_cells)
        self._symbol_queue.expand_and_diffuse(self.cells)

    def slice_to_dict(self, start: int, end: int):
        cells = self.cells[start:end]
        return {
            'cells': [s.get_state() for s in cells],
            'prv': {'n': self._prv.n, 'prp_type': self._prv.prp_type, 'key': list(self._prv.key) if self._prv.key else None},
            'mask_seed': self._mask_hasher.seed,
            'diffusion_seed': self._diffusion_seed,
            'start': start,
            'end': end,
        } 

    def _check_params_dict_compatibility(self, params_dict):
        expected_prv = {'n': self._prv.n, 'prp_type': self._prv.prp_type, 'key': list(self._prv.key) if self._prv.key else None}
        if params_dict['prv'] != expected_prv:
            raise ValueError("Cannot expand from a slice with a different PRV.")
        if params_dict['mask_seed'] != self._mask_hasher.seed:
            raise ValueError("Cannot expand from a slice with a different mask seed.")
        if params_dict['diffusion_seed'] != self._diffusion_seed:
            raise ValueError("Cannot expand from a slice with a different diffusion seed.")

    # 从外部读入一个 cells 列表，并扩展到当前的 cells 列表
    def expand_from_slice(self, slice_dict: Dict[str, Any]):
        self._check_params_dict_compatibility(slice_dict)
        if len(slice_dict['cells']) <= 0: 
            raise ValueError("Cannot expand from an empty slice.")
        if slice_dict['start'] < 0 or slice_dict['end'] < 0 or slice_dict['start'] >= slice_dict['end']:
            raise ValueError("Invalid slice indices.")
        if slice_dict['end'] > len(self.cells):
            self.cells.extend([NNSymbol(self._prv.GF) for _ in range(slice_dict['end'] - len(self.cells))])
        for i in range(slice_dict['start'], slice_dict['end']):
            self.cells[i] += NNSymbol(self._prv.GF, **slice_dict['cells'][i - slice_dict['start']])
        self._symbol_queue.expand_and_diffuse(self.cells)

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
            item = symbol_to_peel.to_item(self._prv)
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
        if not all(c.is_empty() for c in self.cells):
            print("Warning: IBLT4NN decoding may be incomplete.")
            # print("Unpeeled cell",
            #     [(
            #         i, 
            #         c.idx_code_sum == 0,
            #         c.mask_sum == 0,
            #         c.weight_sum == 0,
            #         self._prv.index(c.idx_code_sum/c.mask_sum),
            #         self._prv.index(c.idx_code_sum/c.mask_sum) in self._decoded_weights,
            #         c.is_pure(self._prv)
            #     ) 
            #     for i,c in enumerate(self.cells) if not c.is_empty()])
        return items_peeled_this_round > 0
    @property
    def decoded_weights(self) -> Dict[int, float]:
        return self._decoded_weights
    def is_fully_decoded(self) -> bool:
        return all(c.is_empty() for c in self.cells)
    def scalar_mul_weight(self, scalar: float):
        """Multiplies all cells by a scalar."""
        for cell in self.cells:
            cell.mul_scalar_weight(scalar)
    def remove(self, item: NNItem):
        raise NotImplementedError("RIBLT4NN is aggregation-only.")
    def __sub__(self, other: FilterBase):
        raise NotImplementedError("RIBLT4NN does not support the `-` operation.")
    def __isub__(self, other: FilterBase):
        raise NotImplementedError("RIBLT4NN does not support the `-=` operation.")