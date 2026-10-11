"""Package-level isolation for Honcho unit tests.

Contract (B8): unit tests in this package make ZERO network requests, even
when a live local Honcho is reachable and ambient production config exists.
A real incident proved unique fixtures are not enough - test messages were
written into the production workspace. The guard below turns any connection
attempt into a hard failure; individually marked tests may opt out with
@pytest.mark.allow_network.
"""

import atexit
import importlib.util
import os
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path

import pytest

# Importing the plugin pulls in Hermes core, which at import time reads $HERMES_HOME/.env and
# config.yaml, creates directories there, and can write config backups. Fixtures run only after
# collection, so point HERMES_HOME at a session tempdir before anything is imported. The marker
# arms Hermes' own guard against opening the production state.db.
_SESSION_HERMES_HOME = tempfile.mkdtemp(prefix="hermes-plugin-honcho-test-home-")
atexit.register(shutil.rmtree, _SESSION_HERMES_HOME, True)
os.environ["HERMES_HOME"] = _SESSION_HERMES_HOME
os.environ["HERMES_TEST_ISOLATION"] = _SESSION_HERMES_HOME

# Hermes imports an installed plugin under a synthetic package name; the tests import this
# checkout the same way, under one fixed name, so `mock.patch("hermes_plugin_honcho.client.X")`
# targets the code under test rather than Hermes' bundled copy at `plugins.memory.honcho`.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_honcho", _PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(_PLUGIN_DIR)]
)
_plugin = importlib.util.module_from_spec(_spec)
sys.modules["hermes_plugin_honcho"] = _plugin
_spec.loader.exec_module(_plugin)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: opt-in marker for tests that intentionally touch the network",
    )
    config.addinivalue_line(
        "markers",
        "expect_network_attempts: test asserts on blocked attempts itself",
    )


_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_BASE_URL")


@pytest.fixture(autouse=True)
def _hermetic_environment(tmp_path, monkeypatch):
    """Run every test against an empty HOME and HERMES_HOME with no ambient Honcho/Hermes env.

    A developer's ~/.hermes/config.yaml, ~/.honcho/config.json or HONCHO_API_KEY would
    otherwise change resolved config (timeouts, hosts, workspaces) and point writes at a
    real workspace.
    """
    for name in list(os.environ):
        if name.startswith(("HONCHO_", "HERMES_")) or name.endswith(_CREDENTIAL_SUFFIXES):
            monkeypatch.delenv(name, raising=False)

    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    for sub in ("sessions", "memories", "skills"):
        (hermes_home / sub).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Pin the host so a test cannot inherit a defaultHost and read the wrong host block.
    monkeypatch.setenv("HERMES_HONCHO_HOST", "hermes")
    # tools/lazy_deps would otherwise pip-install a missing feature dependency.
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("TZ", "UTC")

    # Process-global Hermes state that outlives a single test.
    secret_scope = sys.modules.get("agent.secret_scope")
    if secret_scope is not None and hasattr(secret_scope, "_MULTIPLEX_ACTIVE"):
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    hermes_state = sys.modules.get("hermes_state")
    if hermes_state is not None and hasattr(hermes_state, "DEFAULT_DB_PATH"):
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_home / "state.db")


@pytest.fixture
def network_attempts():
    """Recorded connection attempts (visible to tests for assertions)."""
    return []


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch, network_attempts):
    """Fail any test in this package that attempts a real socket connection.

    Recording + teardown assert (not just raising) catches attempts even when
    intermediate code swallows the exception (the async writer retries and
    logs errors instead of propagating them).
    """
    if request.node.get_closest_marker("allow_network"):
        yield
        return

    def _blocked_connect(self, address, *args, **kwargs):
        network_attempts.append(address)
        raise RuntimeError(f"network disabled in honcho unit tests: {address!r}")

    def _blocked_create_connection(address, *args, **kwargs):
        network_attempts.append(address)
        raise RuntimeError(f"network disabled in honcho unit tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)
    yield
    leftover = list(network_attempts)
    if request.node.get_closest_marker("expect_network_attempts"):
        return
    assert not leftover, (
        f"unit test attempted network connections: {leftover} - "
        "inject a fake client/factory before constructing the manager"
    )


@pytest.fixture(autouse=True)
def _no_leaked_writer_threads():
    """Every async manager must be shut down by the test that created it."""
    yield
    leaked = [
        t for t in threading.enumerate()
        if t.name == "honcho-async-writer" and t.is_alive()
    ]
    assert not leaked, (
        f"leaked honcho-async-writer threads: {len(leaked)} - "
        "call manager.shutdown() (use the make_manager fixture)"
    )


# --- hosts.<host>.socketPath tests (test_socket_path.py) -----------------------------------------
# They load the plugin under its own package name and run real Unix-socket servers plus a TCP
# "canary" listener, so that module opts out of the network guard (pytestmark allow_network).

SOCKET_PACKAGE = "hermes_plugin_honcho_under_test"


@pytest.fixture(scope="session")
def plugin():
    try:
        import hermes_cli  # noqa: F401
    except ImportError:
        pytest.skip("Hermes Agent is not importable; put a Hermes checkout on PYTHONPATH")
    if SOCKET_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(SOCKET_PACKAGE, _PLUGIN_DIR / "__init__.py",
                                                      submodule_search_locations=[str(_PLUGIN_DIR)])
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[SOCKET_PACKAGE] = module
        spec.loader.exec_module(module)
    return sys.modules[SOCKET_PACKAGE]


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
    mod = sys.modules[f"{SOCKET_PACKAGE}.client"]
    mod.reset_honcho_client()
    yield mod
    mod.close_honcho_clients()
    mod.reset_honcho_client()


@pytest.fixture
def cache_mod(plugin):
    return sys.modules[f"{SOCKET_PACKAGE}.client_cache"]
