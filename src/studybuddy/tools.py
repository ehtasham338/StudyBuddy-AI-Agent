"""
HANDS-ON TOOLS: Calculator — the agent's pocket calculator.
This module contains only the tools that the agent can use itself.
"""

import ast
import operator

from langchain_core.tools import tool


# Only these mathematical operations are allowed.
# All other operations are blocked.
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate_expression(node):
    """
    Everything else is blocked.
    Only numbers and the allowed mathematical operations are permitted.
    """

    # Check whether the node contains a number.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # Handle binary operations such as +, -, *, and /.
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](
            _calculate_expression(node.left),
            _calculate_expression(node.right)
        )

    # Handle unary operations such as -5 and +5.
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](
            _calculate_expression(node.operand)
        )

    # Reject unsupported expressions.
    raise ValueError(
        "Only mathematical expressions are allowed: "
        "numbers and + - * / // % ** ( )"
    )


@tool
def calculate(expression: str) -> str:
    """
    Solve an arithmetic expression.

    Input must contain only a mathematical expression,
    such as '2 + 2' or '800 * 15 / 100'.
    """

    try:
        # Convert the expression into an Abstract Syntax Tree (AST).
        tree = ast.parse(expression, mode="eval")

        # Evaluate the expression and return the result as a string.
        return str(_calculate_expression(tree.body))

    except Exception as e:
        return f"Error: Unable to evaluate this expression ({e})"


if __name__ == "__main__":

    # Test valid mathematical expressions.
    for expr in [
        "2 + 2",
        "800 * 15 / 100",
        "2 ** 10",
        "(2 + 3) * 4",
        "15 % 4"
    ]:
        print(f"{expr:20} -> {calculate.invoke({'expression': expr})}")

    # Test invalid expressions and error handling.
    print("--- Security checks (these should return errors) ---")

    for expr in [
        "__import__('os').system('ls')",
        "2 / 0"
    ]:
        print(
            f"{expr:35} -> "
            f"{calculate.invoke({'expression': expr})[:45]}..."
        )