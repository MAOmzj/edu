import unittest

from utils.math_tool import safe_calculate


class SafeCalculateTests(unittest.TestCase):
    def test_basic_operations(self):
        self.assertEqual(safe_calculate("(25 + 15) * 3"), "120")
        self.assertEqual(safe_calculate("7 ÷ 2"), "3.5")
        self.assertEqual(safe_calculate("2^3 + 1"), "9")
        self.assertEqual(safe_calculate("10 % 3"), "1")

    def test_friendly_errors(self):
        for expression in ("", "1 / 0", "__import__('os')", "2 ** 9"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    safe_calculate(expression)


if __name__ == "__main__":
    unittest.main()
