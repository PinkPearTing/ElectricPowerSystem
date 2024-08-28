import unittest

# 发现所有测试用例
loader = unittest.TestLoader()
tests = loader.discover('./')

# 创建测试运行器
runner = unittest.TextTestRunner()

# 运行测试用例并展示结果
result = runner.run(tests)