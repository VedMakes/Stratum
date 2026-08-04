from __future__ import annotations

import importlib
import sys
from pathlib import Path


def core_train():
    """Resolve the existing MLProj API in editable and installed layouts."""
    try:
        return importlib.import_module("MLProj.api").train
    except ModuleNotFoundError:
        package_root = Path(__file__).resolve().parents[1]
        parent = str(package_root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return importlib.import_module(f"{package_root.name}.api").train


def core_policy():
    """Resolve the core training-policy function."""
    try:
        return importlib.import_module("MLProj.policy").resolve_training_policy
    except ModuleNotFoundError:
        package_root = Path(__file__).resolve().parents[1]
        parent = str(package_root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return importlib.import_module(f"{package_root.name}.policy").resolve_training_policy
