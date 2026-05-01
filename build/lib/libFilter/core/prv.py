# libFilter/core/prv.py

"""
Implementation of a "Wunderbar" Pseudo-Random Vector (PRV) using fixed
cryptographic primitives.

This PRV supports efficient forward lookup (entry) and reverse lookup (index)
over a large virtual vector of Galois Field elements. It is based on a
Pseudo-Random Permutation (PRP) like AES or DES.
"""

import hashlib
from typing import Dict, Any, Optional

import galois
from Crypto.Cipher import AES, DES

# TODO:
# Consider introducing the `ffex` library to implement Format-Preserving Encryption (FPE),
# enabling same-bit-width encryption mappings (e.g., 48-bit to 48-bit, 40-bit to 40-bit, etc.).

class PRV:
    """
    A Pseudo-Random Vector that maps indices to finite field elements.

    The mapping is determined by a cryptographic permutation (PRP) and a
    pre-defined configuration.
    """
    
    # Configuration presets for different PRP types.
    CONFIGS: Dict[str, Dict[str, Any]] = {
        'aes128': {
            'bits': 128,
            'key_size': 16,
            'cipher': AES,
            'p_offset': 51, # For GF(2^128 + 51)
            'seed': "prv-aes128-default-seed"
        },
        'des64':  {
            'bits': 64,
            'key_size': 8,
            'cipher': DES,
            'p_offset': 13, # For GF(2^64 + 13)
            'seed': "prv-des64-default-seed"
        }
    }

    def __init__(self, n: int, prp_type: str = 'aes128', key: Optional[Any] = None):
        """
        Initializes the Pseudo-Random Vector.

        Args:
            n: The virtual length of the vector.
            prp_type: The PRP preset type ('aes128' or 'des64').
            key: A custom cryptographic key (bytes or str). If None, a deterministic key is
                 generated from a fixed seed.
        """
        config_key = prp_type.lower()
        if config_key not in self.CONFIGS:
            raise ValueError(f"Unsupported PRP type. Choose from {list(self.CONFIGS.keys())}.")
        
        config = self.CONFIGS[config_key]
        key_size = config['key_size']

        self.n = n
        self.prp_type = config_key
        self.prp_bits = config['bits']
        
        # Define the prime for the Galois Field
        self.p = (1 << self.prp_bits) + config['p_offset']
        
        if key is not None:
            # Accept str or bytes for key, convert to bytes if needed
            if isinstance(key, str):
                key_bytes = key.encode()
            elif isinstance(key, bytes):
                key_bytes = key
            else:
                raise TypeError(f"Key must be bytes or str, got {type(key)}")
            # Truncate or pad the key to the required key_size
            if len(key_bytes) < key_size:
                key_bytes = key_bytes.ljust(key_size, b'\0')
            elif len(key_bytes) > key_size:
                key_bytes = key_bytes[:key_size]
            self.key = key_bytes
        else:
            # Generate a deterministic key from the seed if none is provided.
            self.key = hashlib.sha256(config['seed'].encode()).digest()[:key_size]
        
        self.block_size_bytes = self.prp_bits // 8
        self.value_limit = 1 << self.prp_bits
        
        # Initialize the cipher in ECB mode, which acts as a PRP.
        self._cipher = config['cipher'].new(self.key, config['cipher'].MODE_ECB)
        self.GF = galois.GF(self.p)
        # print(self.GF)

    def entry(self, i: int) -> galois.FieldArray:
        """Gets the field element at index `i`."""
        if not 0 <= i < self.n:
            raise IndexError(f"Index {i} is out of the valid range [0, {self.n-1}]")
            
        # Permute the index to get the pseudo-random value.
        b = i.to_bytes(self.block_size_bytes, 'big')
        b_prime = self._cipher.encrypt(b)
        return self.GF(int.from_bytes(b_prime, 'big'))

    def index(self, k: Any) -> Optional[int]:
        """
        Finds the index corresponding to a given field element `k`.
        Returns None if the value is invalid or maps to an out-of-bounds index.
        """
        try:
            k_int = int(k)
        except (ValueError, TypeError):
            return None

        # The value must be within the PRP's domain [0, 2^bits - 1].
        if not 0 <= k_int < self.value_limit:
            return None
            
        # Reverse the permutation to find the original index.
        b_prime = k_int.to_bytes(self.block_size_bytes, 'big')
        b = self._cipher.decrypt(b_prime)
        i = int.from_bytes(b, 'big')
        
        # The result is valid only if the decoded index is within our virtual vector length `n`.
        return i if i < self.n else None

    def __repr__(self) -> str:
        return f"PRV(n={self.n}, prp_type='{self.prp_type}')"