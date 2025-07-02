import hashlib
import galois
import sys
import unittest
import argparse
from Crypto.Cipher import AES, DES

class PRV:
    """
    一个 "Wunderbar" 伪随机向量的实现，使用固定的密码学参数。

    该向量支持高效的元素访问 (entry) 和反向索引查找 (index)。
    PRP类型 ('aes128' 或 'des64') 决定了内部使用的密钥和有限域。
    """
    CONFIGS = {
        'aes128': {'bits': 128, 'key_size': 16, 'cipher': AES, 'p_offset': 51, 'seed': "prv-aes128-seed"},
        'des64':  {'bits': 64,  'key_size': 8,  'cipher': DES, 'p_offset': 13, 'seed': "prv-des64-seed"}
    }

    def __init__(self, n: int, prp_type: str = 'aes128', key: bytes | None = None):
        """
        初始化伪随机向量。

        Args:
            n (int): 向量的虚拟长度。
            prp_type (str): PRP预设类型 ('aes128' 或 'des64')。
            key (bytes, optional): 自定义密钥。若为 None，则从固定种子生成确定性密钥。
        """
        config_key = prp_type.lower()
        if config_key not in self.CONFIGS:
            raise ValueError(f"不支持的 PRP 类型。请选择 {list(self.CONFIGS.keys())}。")
        
        config = self.CONFIGS[config_key]
        key_size = config['key_size']

        self.n = n
        self.prp_type = config_key
        self.prp_bits = config['bits']
        self.p = (1 << self.prp_bits) + config['p_offset']
        
        if key is not None:
            if len(key) != key_size:
                raise ValueError(f"'{prp_type}' 需要 {key_size} 字节的密钥，但提供了 {len(key)} 字节。")
            self.key = key
        else:
            self.key = hashlib.sha256(config['seed'].encode()).digest()[:key_size]
        
        self.block_size_bytes = self.prp_bits // 8
        self.value_limit = 1 << self.prp_bits
        self._cipher = config['cipher'].new(self.key, config['cipher'].MODE_ECB)
        self.GF = galois.GF(self.p)

    def entry(self, i: int) -> galois.FieldArray:
        """获取向量在索引 i 处的域元素值。"""
        if not 0 <= i < self.n:
            raise IndexError("索引超出范围")
        b = i.to_bytes(self.block_size_bytes, 'big')
        b_prime = self._cipher.encrypt(b)
        return self.GF(int.from_bytes(b_prime, 'big'))

    def index(self, k) -> int | None:
        """根据域元素值 k 查找其索引。"""
        try: k_int = int(k)
        except (ValueError, TypeError): return None

        if not 0 <= k_int < self.value_limit:
            return None
            
        b_prime = k_int.to_bytes(self.block_size_bytes, 'big')
        b = self._cipher.decrypt(b_prime)
        i = int.from_bytes(b, 'big')
        return i if i < self.n else None

    def __repr__(self):
        return f"PRV(n={self.n}, prp_type='{self.prp_type}')"

# --- In-module Tests ---
class TestPRV(unittest.TestCase):
    def setUp(self):
        """为每个测试设置共享参数。"""
        self.n = 100_000
        self.prp_types = list(PRV.CONFIGS.keys())

    def test_determinism_with_default_key(self):
        """测试默认密钥的确定性。"""
        for prp_type in self.prp_types:
            with self.subTest(prp_type=prp_type):
                prv1 = PRV(n=self.n, prp_type=prp_type)
                prv2 = PRV(n=self.n, prp_type=prp_type)
                self.assertEqual(prv1.key, prv2.key)

    def test_user_provided_key(self):
        """测试使用自定义密钥的功能。"""
        for prp_type in self.prp_types:
            with self.subTest(prp_type=prp_type):
                key_size = PRV.CONFIGS[prp_type]['key_size']
                
                # 测试有效密钥
                custom_key = b'\xAA' * key_size
                prv = PRV(n=self.n, prp_type=prp_type, key=custom_key)
                self.assertEqual(prv.key, custom_key)
                
                # 测试无效密钥
                invalid_key = b'\xBB' * (key_size + 1)
                with self.assertRaises(ValueError):
                    PRV(n=self.n, prp_type=prp_type, key=invalid_key)

    def test_forward_and_backward_conversion(self):
        """测试 entry() 和 index() 的双向转换。"""
        for prp_type in self.prp_types:
            with self.subTest(prp_type=prp_type):
                prv = PRV(n=self.n, prp_type=prp_type)
                test_index = 12345
                gf_element = prv.entry(test_index)
                found_index = prv.index(gf_element)
                self.assertIsInstance(gf_element, galois.FieldArray)
                self.assertEqual(found_index, test_index)

    def test_galois_field_arithmetic(self):
        """测试返回的域元素是否支持正确的算术运算。"""
        for prp_type in self.prp_types:
            with self.subTest(prp_type=prp_type):
                prv = PRV(n=self.n, prp_type=prp_type)
                val1, val2 = prv.entry(9876), prv.entry(5432)
                self.assertEqual(val1 + val2, prv.GF(int(val1)) + prv.GF(int(val2)))
                self.assertEqual(val1 * val2, prv.GF(int(val1)) * prv.GF(int(val2)))

    def test_edge_cases(self):
        """测试边界条件和无效输入。"""
        for prp_type in self.prp_types:
            with self.subTest(prp_type=prp_type):
                prv = PRV(n=self.n, prp_type=prp_type)
                with self.assertRaises(IndexError):
                    prv.entry(self.n)
                
                invalid_k = 1 << prv.prp_bits
                self.assertIsNone(prv.index(invalid_k))
                self.assertIsNone(prv.index("invalid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PRV tests.")
    cli_args, unknown = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0]] + unknown, verbosity=2, exit=False)