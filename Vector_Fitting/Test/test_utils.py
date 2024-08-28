import numpy as np


def assert_dicts_equal(dict1, dict2, rtol=1e-7, atol=1e-10):
    """
    比较两个字典是否相等,包括处理嵌套的三维矩阵和浮点数。

    Args:
        dict1 (dict): 第一个字典
        dict2 (dict): 第二个字典
        rtol (float): 浮点数比较的相对容差
        atol (float): 浮点数比较的绝对容差

    Raises:
        AssertionError: 如果两个字典不相等
    """

    assert set(dict1.keys()) == set(dict2.keys()), "字典的键不相同"

    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            # 如果是NumPy数组,则使用np.allclose进行比较
            assert np.allclose(val1, val2, rtol=rtol, atol=atol), f"键 {key} 对应的值不相等"
        else:
            # 否则直接进行值比较
            assert val1 == val2, f"键 {key} 对应的值不相等"