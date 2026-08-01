"""不使用 eval 的基础算术表达式计算器。"""

import ast
import math
import operator


MAX_EXPRESSION_LENGTH = 120
MAX_ABSOLUTE_VALUE = 1_000_000_000_000
MAX_POWER = 8

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_result(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("表达式中只能使用数字")
    if not math.isfinite(float(value)):
        raise ValueError("计算结果不是有限数")
    if abs(value) > MAX_ABSOLUTE_VALUE:
        raise ValueError("计算结果过大")
    return value


def _evaluate(node: ast.AST, depth: int = 0) -> int | float:
    if depth > 20:
        raise ValueError("表达式嵌套过深")

    if isinstance(node, ast.Expression):
        return _evaluate(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        return _check_result(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _evaluate(node.operand, depth + 1)
        return _check_result(_UNARY_OPERATORS[type(node.op)](value))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left, depth + 1)
        right = _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
            raise ValueError(f"指数不能超过 {MAX_POWER}")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ValueError("除数不能为 0") from exc
        return _check_result(result)

    raise ValueError("只支持数字、括号和 + - × ÷ // % ** 运算")


def safe_calculate(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("算式不能为空")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("算式过长")

    normalized = (
        expression.strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("算式格式不正确") from exc

    result = _evaluate(tree)
    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))
        return format(result, ".12g")
    return str(result)
