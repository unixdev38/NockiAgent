"""Calculator tool: arithmetic evaluation through an AST allow-list.

There is no `eval()` here — expressions are parsed with `ast` and evaluated by
a visitor that only accepts a fixed set of nodes (arithmetic operators,
numeric constants, whitelisted `math` functions and a few builtins). Anything
else (names, attributes, imports, subscripts, comprehensions, ...) raises
`SafeEvalError`.
"""

from __future__ import annotations

import ast
import math
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.security.permissions import Permission
from app.tools.base import Tool, ToolContext, ToolResult


class SafeEvalError(ValueError):
    """Raised when an expression uses forbidden syntax or names."""


_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log1p": math.log1p,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "expm1": math.expm1,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "fabs": math.fabs,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "gcd": math.gcd,
    "factorial": math.factorial,
    "pow": pow,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_ALLOWED_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class _SafeEvaluator(ast.NodeVisitor):
    """Evaluate an AST using an explicit allow-list of constructs."""

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise SafeEvalError(f"Opérateur interdit: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _BINARY_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise SafeEvalError(f"Opérateur unaire interdit: {op_type.__name__}")
        return _UNARY_OPS[op_type](self.visit(node.operand))

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise SafeEvalError(f"Constante interdite: {node.value!r}")
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise SafeEvalError(f"Nom interdit: {node.id!r}")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            described = getattr(node.func, "id", type(node.func).__name__)
            raise SafeEvalError(f"Fonction interdite: {described!r}")
        if node.keywords:
            raise SafeEvalError("Arguments nommés interdits")
        args = [self.visit(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    def generic_visit(self, node: ast.AST) -> Any:
        raise SafeEvalError(f"Syntaxe interdite: {type(node).__name__}")


def safe_eval(expression: str) -> int | float:
    """Safely evaluate an arithmetic expression. Raises `SafeEvalError`."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise SafeEvalError(f"Expression invalide: {exc}") from exc
    try:
        result = _SafeEvaluator().visit(tree)
    except SafeEvalError:
        raise
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as exc:
        raise SafeEvalError(f"Calcul impossible: {exc}") from exc
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise SafeEvalError(f"Résultat non numérique: {result!r}")
    return result


class CalculatorInput(BaseModel):
    expression: str = Field(description="Expression arithmétique, ex: '2 + 2 * 10' ou 'sqrt(16)'")


class CalculatorTool(Tool):
    """Evaluate arithmetic safely (AST allow-list, no `eval`)."""

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Évalue une expression arithmétique (opérateurs + - * / // % **, "
        "fonctions mathématiques usuelles: sqrt, sin, cos, log, ...). "
        "Exemple: 'sqrt(144) + 2^…' → utiliser ** pour la puissance."
    )
    permission: ClassVar[Permission] = Permission.READ
    args_schema: ClassVar[type[BaseModel]] = CalculatorInput

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)

    def run(self, expression: str = "", **_: Any) -> ToolResult:
        try:
            result = safe_eval(expression)
        except SafeEvalError as exc:
            return self._err(str(exc))
        rendered = f"{result}"
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            rendered = f"{result} (= {int(result)})"
        return self._ok(rendered)


def build_calculator_tool(context: ToolContext) -> CalculatorTool:
    return CalculatorTool(context)
