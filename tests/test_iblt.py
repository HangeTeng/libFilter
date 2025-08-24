# tests/test_iblt.py

"""IBLT4NN（神经网络用IBLT）的测试。"""

import pytest
import math

from libFilter.core.prv import PRV
from libFilter.core.utils import HashMapping
from libFilter.filters4nn.iblt4nn import IBLT4NN
from libFilter.filters4nn.nn_utils import NNItem

@pytest.fixture
def nn_setup():
    """为IBLT4NN测试提供标准设置。"""
    n_indices = 2048
    hash_map = HashMapping.from_seeds(['NNS1', 'NNS2', 'NNS3'], table_size=50)
    prv = PRV(n=n_indices, prp_type='aes128')
    client1_updates = [NNItem(idx=10, weight=0.5), NNItem(idx=1024, weight=-0.1)]
    client2_updates = [NNItem(idx=10, weight=0.2), NNItem(idx=100, weight=0.3)]
    return {
        'hash_map': hash_map,
        'prv': prv,
        'client1': client1_updates,
        'client2': client2_updates,
    }

def test_aggregation_and_peel(nn_setup):
    """测试IBLT聚合和剥离(peel)功能。"""
    map_config, prv = nn_setup['hash_map'], nn_setup['prv']
    client1, client2 = nn_setup['client1'], nn_setup['client2']

    iblt = IBLT4NN(map_config, prv)
    for item in client1:
        iblt.push(item)
    for item in client2:
        iblt.push(item)

    # 非破坏性peel
    decoded = iblt.peel(destructive=False)
    expected = {10: 0.7, 100: 0.3, 1024: -0.1}
    assert len(decoded) == len(expected)
    for idx, weight in expected.items():
        assert idx in decoded
        assert math.isclose(decoded[idx], weight)

    # 破坏性peel后应为空
    iblt2 = iblt.copy()
    iblt2.peel(destructive=True)
    assert iblt2.is_fully_decoded()

def test_merge_and_peel(nn_setup):
    """测试IBLT的合并与peel。"""
    map_config, prv = nn_setup['hash_map'], nn_setup['prv']
    client1, client2 = nn_setup['client1'], nn_setup['client2']

    iblt1 = IBLT4NN(map_config, prv)
    for i in client1:
        iblt1.push(i)
    iblt2 = IBLT4NN(map_config, prv)
    for i in client2:
        iblt2.push(i)

    merged = iblt1 + iblt2
    decoded = merged.peel()
    expected = {10: 0.7, 100: 0.3, 1024: -0.1}
    assert len(decoded) == len(expected)
    for idx, weight in expected.items():
        assert idx in decoded
        assert math.isclose(decoded[idx], weight)

    # += 操作
    iblt1_copy = iblt1.copy()
    iblt1_copy += iblt2
    assert iblt1_copy.to_dict() == merged.to_dict()

def test_serialization(nn_setup):
    """测试IBLT的序列化和反序列化。"""
    map_config, prv = nn_setup['hash_map'], nn_setup['prv']
    iblt = IBLT4NN(map_config, prv)
    for item in nn_setup['client1']:
        iblt.push(item)
    config = iblt.to_dict()
    rebuilt = IBLT4NN.from_dict(config)
    assert iblt.to_dict() == rebuilt.to_dict()
    assert rebuilt.peel() == iblt.peel()

def test_unsupported_operations(nn_setup):
    """测试不支持的操作会抛出NotImplementedError。"""
    iblt = IBLT4NN(nn_setup['hash_map'], nn_setup['prv'])
    with pytest.raises(NotImplementedError, match="does not support `remove`"):
        iblt.remove(nn_setup['client1'][0])
    with pytest.raises(NotImplementedError, match="does not support the `-` operation"):
        _ = iblt - iblt
    with pytest.raises(NotImplementedError, match="does not support the `-=` operation"):
        iblt -= iblt