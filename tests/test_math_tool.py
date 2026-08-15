# 文件用途：验证安全计算器支持的基础运算和对危险/超限表达式的拒绝。
# 调用关系：unittest 调用本文件；本文件直接调用 utils/math_tool.py。
# 修改易踩坑：新增合法语法时仍要保留代码执行、除零、超大指数和空输入测试。
import unittest

from utils.math_tool import safe_calculate


class SafeCalculateTests(unittest.TestCase):
    def test_basic_operations(self):
        """验证计算器能够处理括号、四则运算、乘方和取余。"""
        self.assertEqual(safe_calculate("(25 + 15) * 3"), "120")
        self.assertEqual(safe_calculate("7 ÷ 2"), "3.5")
        self.assertEqual(safe_calculate("2^3 + 1"), "9")
        self.assertEqual(safe_calculate("10 % 3"), "1")

    def test_friendly_errors(self):
        """验证空算式、除零、危险代码和超限指数会被友好拒绝。"""
        for expression in ("", "1 / 0", "__import__('os')", "2 ** 9"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    safe_calculate(expression)


if __name__ == "__main__":
    unittest.main()
