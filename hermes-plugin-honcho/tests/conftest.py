"""Fixtures for the plugin tests.

The plugin imports Hermes core modules (``agent``, ``hermes_cli``, ``plugins.plugin_utils``), so run these
tests with a Hermes Agent source tree on ``PYTHONPATH`` and with its venv, from the repository root:

    PYTHONPATH=~/.hermes/hermes-agent ~/.hermes/hermes-agent/venv/bin/python -m pytest \\
        hermes-plugin-honcho/tests -o addopts= -o pythonpath= --import-mode=importlib

``-o pythonpath=`` keeps the server's ``src/`` off the path (its ``utils`` package shadows Hermes'), and
``--import-mode=importlib`` is needed because the plugin directory name is not a valid package name.
The plugin is loaded under its own package name, the way Hermes loads a user-installed provider.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE = "hermes_plugin_honcho_under_test"


@pytest.fixture(scope="session")
def plugin():
    try:
        import hermes_cli  # noqa: F401
    except ImportError:
        pytest.skip("Hermes Agent is not importable; put a Hermes checkout on PYTHONPATH")
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, PLUGIN_DIR / "__init__.py",
                                                      submodule_search_locations=[str(PLUGIN_DIR)])
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    return sys.modules[PACKAGE]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("HONCHO_API_KEY", "HONCHO_BASE_URL", "HONCHO_URL", "HONCHO_TIMEOUT", "HERMES_HONCHO_HOST",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY",
                "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def client_mod(plugin, hermes_home):
    mod = sys.modules[f"{PACKAGE}.client"]
    mod.reset_honcho_client()
    yield mod
    mod.close_honcho_clients()
    mod.reset_honcho_client()


@pytest.fixture
def cache_mod(plugin):
    return sys.modules[f"{PACKAGE}.client_cache"]
