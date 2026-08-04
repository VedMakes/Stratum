from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Call, ast.Attribute,
}
ALLOWED_NUMPY = {"log1p", "sqrt", "abs", "maximum", "minimum", "clip"}


def validate_formula(formula: str, columns: list[str]) -> ast.Expression:
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError(f"Unsupported formula operation: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in set(columns) | {"np"}:
            raise ValueError(f"Unknown formula name: {node.id}")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "np" or node.attr not in ALLOWED_NUMPY:
                raise ValueError("Only approved numpy functions are allowed")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Attribute):
            raise ValueError("Direct function calls are not allowed")
    return tree


def evaluate_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    tree = validate_formula(formula, [str(col) for col in df.columns])
    env: dict[str, Any] = {str(col): df[col] for col in df.columns}
    env["np"] = np
    value = eval(compile(tree, "<feature-formula>", "eval"), {"__builtins__": {}}, env)
    if np.isscalar(value):
        value = pd.Series([value] * len(df), index=df.index)
    value = pd.Series(value, index=df.index)
    if len(value) != len(df):
        raise ValueError("Formula result length does not match dataset")
    return value
