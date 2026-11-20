import tomllib
from pathlib import Path
import subprocess
import sys


def _project():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_exposes_mcp_script_and_packages():
    config = _project()
    assert config["project"]["scripts"]["stratum-mcp"] == "stratum.mcp_server:main"
    assert config["project"]["scripts"]["stratum"] == "stratum.cli:main"
    packages = set(config["tool"]["setuptools"]["packages"])
    assert {"MLProj", "MLProj.statistical", "stratum", "stratum.data", "stratum.tools"} <= packages
    assert config["tool"]["setuptools"]["package-dir"]["MLProj"] == "."


def test_pyproject_has_test_and_llm_extras():
    config = _project()
    extras = config["project"]["optional-dependencies"]
    assert "pytest>=8.0" in extras["test"]
    assert any(dep.startswith("openai") for dep in extras["llm-openai"])
    assert any(dep.startswith("anthropic") for dep in extras["llm-anthropic"])
    assert "mcp>=1.27,<2" in extras["stratum"]
    assert "imbalanced-learn>=0.14.1" in extras["smote"]
    assert "catboost>=1.2.10" not in config["project"]["dependencies"]


def test_core_registry_import_does_not_require_optional_model_or_viz_packages():
    code = r'''import importlib.abc
import sys
blocked = {"catboost", "lightgbm", "xgboost", "matplotlib", "shap", "seaborn", "dtreeviz"}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
import MLProj
from MLProj.registry import create_model
create_model("random_forest", "classification")
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_pytest_can_import_root_package_from_repo():
    config = _project()
    assert config["tool"]["pytest"]["ini_options"]["pythonpath"] == [".."]
