import tomllib
from pathlib import Path


def _project():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_exposes_mcp_script_and_packages():
    config = _project()
    assert config["project"]["scripts"]["stratum-mcp"] == "stratum.mcp_server:main"
    packages = set(config["tool"]["setuptools"]["packages"])
    assert {"MLProj", "MLProj.statistical", "stratum", "stratum.data", "stratum.tools"} <= packages
    assert config["tool"]["setuptools"]["package-dir"]["MLProj"] == "."


def test_pyproject_has_test_and_llm_extras():
    extras = _project()["project"]["optional-dependencies"]
    assert "pytest>=8.0" in extras["test"]
    assert any(dep.startswith("openai") for dep in extras["llm-openai"])
    assert any(dep.startswith("anthropic") for dep in extras["llm-anthropic"])


def test_pytest_can_import_root_package_from_repo():
    config = _project()
    assert config["tool"]["pytest"]["ini_options"]["pythonpath"] == [".."]
