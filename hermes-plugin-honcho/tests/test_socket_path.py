"""hosts.<host>.socketPath: Unix-socket transport through the SDK's public ``http_client``.

Every network test runs a real HTTP server on a Unix socket plus a TCP "canary" listener that must
never see a connection: the canary is the base_url host, and in one test also the proxy from env.
"""

from __future__ import annotations

import gc
import json
import os
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

# --- helpers ----------------------------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def _reply(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.server.seen.append({"method": self.command, "path": self.path, "host": self.headers.get("Host"),
                                 "auth": self.headers.get("Authorization")})
        if self.server.hang:
            time.sleep(self.server.hang)
            return
        body = json.dumps({"id": "ws", "metadata": {"k": "v"}, "configuration": {},
                           "created_at": "2026-01-01T00:00:00Z"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_PUT = _reply

    def log_message(self, *args):
        pass


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):  # BaseHTTPRequestHandler expects (host, port) client_address
        request, _ = super().get_request()
        return request, ("uds", 0)


class UdsServer:
    def __init__(self, path, hang: float = 0.0):
        self.path = str(path)
        self.server = _UnixHTTPServer(self.path, _Handler)
        self.server.seen, self.server.hang = [], hang
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def seen(self):
        return self.server.seen

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        if os.path.exists(self.path):
            os.unlink(self.path)


class TcpCanary:
    """Counts every TCP connection; the socket transport must never open one."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.sock.settimeout(0.1)
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self.connections += 1
            conn.close()

    def close(self):
        self._stop = True
        self.thread.join(timeout=2)
        self.sock.close()


@pytest.fixture
def canary():
    c = TcpCanary()
    yield c
    c.close()


@pytest.fixture
def sock_dir(tmp_path):
    # AF_UNIX paths are limited to ~107 bytes; pytest tmp paths can exceed that.
    import tempfile
    d = tempfile.mkdtemp(prefix="hsk-")
    yield d
    for name in os.listdir(d):
        os.unlink(os.path.join(d, name))
    os.rmdir(d)


def write_honcho_json(home, host_block: dict, **root):
    (home / "honcho.json").write_text(json.dumps({**root, "hosts": {"hermes": host_block}}))


def make_config(client_mod, **overrides):
    cfg = client_mod.HonchoClientConfig.from_global_config(host="hermes")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --- config parsing / validation -----------------------------------------------------------------------

def test_socket_path_is_read_from_host_block(client_mod, hermes_home):
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/h.sock"})
    assert make_config(client_mod).socket_path == "/run/x/h.sock"


@pytest.mark.parametrize("block", [{}, {"socketPath": ""}, {"socketPath": None}])
def test_unset_socket_path_is_none(client_mod, hermes_home, block):
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", **block})
    assert make_config(client_mod).socket_path is None


def test_socket_path_is_not_inherited_from_root(client_mod, hermes_home):
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k"}, socketPath="/run/x/h.sock")
    cfg = make_config(client_mod)
    assert cfg.socket_path is client_mod._ROOT_SOCKET_PATH
    with pytest.raises(ValueError, match="host-local"):
        client_mod._build_client(cfg)


def test_other_host_does_not_see_socket_path(client_mod, hermes_home):
    (hermes_home / "honcho.json").write_text(json.dumps({"hosts": {
        "hermes": {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/a.sock"},
        "hermes_b": {"baseUrl": "http://localhost", "apiKey": "k2"}}}))
    cfg = client_mod.HonchoClientConfig.from_global_config(host="hermes_b")
    assert cfg.socket_path is None


@pytest.mark.parametrize("value", ["relative/h.sock", "./h.sock", "@abstract", "/run/x\x00y", 5, ["/a"], " /a"])
def test_malformed_socket_path_rejected(client_mod, hermes_home, value, monkeypatch):
    built = []
    monkeypatch.setattr("honcho.Honcho", lambda **kw: built.append(kw))
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": value})
    with pytest.raises(ValueError, match="absolute filesystem path"):
        client_mod._build_client(make_config(client_mod))
    assert built == []  # never fell back to a TCP client


@pytest.mark.parametrize("base_url", [None, "https://localhost", "http://api.honcho.dev", "http://8.8.8.8:8000"])
def test_socket_path_requires_local_http_base_url(client_mod, hermes_home, base_url, monkeypatch):
    built = []
    monkeypatch.setattr("honcho.Honcho", lambda **kw: built.append(kw))
    block = {"apiKey": "k", "socketPath": "/run/x/h.sock"}
    if base_url:
        block["baseUrl"] = base_url
    write_honcho_json(hermes_home, block)
    with pytest.raises(ValueError, match="local http baseUrl"):
        client_mod._build_client(make_config(client_mod))
    assert built == []


def test_socket_path_with_oauth_rejected(client_mod, hermes_home):
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/h.sock",
                                    "oauth": {"refreshToken": "r"}})
    with pytest.raises(ValueError, match="OAuth"):
        client_mod._build_client(make_config(client_mod))


def test_provider_init_fails_closed_on_bad_socket_path(plugin, client_mod, hermes_home, canary):
    """The real provider lifecycle: a malformed socketPath leaves memory off, with no TCP attempt."""
    write_honcho_json(hermes_home, {"baseUrl": f"http://127.0.0.1:{canary.port}", "apiKey": "k",
                                    "socketPath": "relative.sock", "peerName": "u", "recallMode": "tools",
                                    "initOnSessionStart": True})
    provider = plugin.HonchoMemoryProvider()
    provider.initialize("s1", platform="cli")
    assert provider._manager is None
    assert canary.connections == 0


# --- old TCP behavior unchanged -------------------------------------------------------------------------

def test_tcp_kwargs_unchanged_without_socket_path(client_mod, hermes_home, monkeypatch):
    seen = []
    monkeypatch.setattr("honcho.Honcho", lambda **kw: seen.append(kw) or object())
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost:8000/v3", "apiKey": "k", "timeout": 7})
    client_mod._build_client(make_config(client_mod))
    assert seen == [{"workspace_id": "hermes", "api_key": "k", "environment": "production", "timeout": 7.0,
                     "default_headers": client_mod.telemetry_headers(), "base_url": "http://localhost:8000"}]
    assert len(client_mod._owned_http_clients) == 0


# --- sync transport, credential, timeout ------------------------------------------------------------------

def test_requests_go_over_the_socket_only(client_mod, hermes_home, sock_dir, canary, monkeypatch):
    srv = UdsServer(os.path.join(sock_dir, "h.sock"))
    try:
        # Proxy env pointing at the canary: trust_env=False must keep it from mounting a TCP proxy.
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{canary.port}")
        monkeypatch.setenv("ALL_PROXY", f"http://127.0.0.1:{canary.port}")
        write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "jwt-a", "workspace": "ws-a",
                                        "socketPath": srv.path, "timeout": 5})
        honcho = client_mod.get_honcho_client(make_config(client_mod))
        assert honcho.get_metadata() == {"k": "v"}
        assert srv.seen and all(r["path"].startswith("/v3/") for r in srv.seen)
        assert {r["auth"] for r in srv.seen} == {"Bearer jwt-a"}
        assert {r["host"] for r in srv.seen} == {"localhost"}
        assert canary.connections == 0
    finally:
        srv.stop()


def test_base_url_host_is_never_dialed(client_mod, hermes_home, sock_dir, canary):
    srv = UdsServer(os.path.join(sock_dir, "h.sock"))
    try:
        write_honcho_json(hermes_home, {"baseUrl": f"http://127.0.0.1:{canary.port}", "apiKey": "k",
                                        "socketPath": srv.path})
        client_mod.get_honcho_client(make_config(client_mod)).get_metadata()
        assert len(srv.seen) >= 1 and canary.connections == 0
    finally:
        srv.stop()


def _errno_names(exc):
    """errno names found along the exception's cause/context chain."""
    import errno
    names, seen = set(), set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno:
            names.add(errno.errorcode.get(exc.errno, str(exc.errno)))
        exc = exc.__cause__ or exc.__context__
    return names


def test_missing_socket_fails_without_fallback(client_mod, hermes_home, sock_dir, canary):
    from honcho.http.exceptions import ConnectionError as HonchoConnectionError
    write_honcho_json(hermes_home, {"baseUrl": f"http://127.0.0.1:{canary.port}", "apiKey": "k",
                                    "socketPath": os.path.join(sock_dir, "absent.sock"), "timeout": 2})
    honcho = client_mod.get_honcho_client(make_config(client_mod))
    with pytest.raises(HonchoConnectionError) as exc:
        honcho.get_metadata()
    assert "ENOENT" in _errno_names(exc.value)
    assert canary.connections == 0


def test_revoked_socket_stops_a_running_client(client_mod, hermes_home, sock_dir, canary):
    from honcho.http.exceptions import ConnectionError as HonchoConnectionError
    srv = UdsServer(os.path.join(sock_dir, "h.sock"))
    write_honcho_json(hermes_home, {"baseUrl": f"http://127.0.0.1:{canary.port}", "apiKey": "k",
                                    "socketPath": srv.path, "timeout": 2})
    honcho = client_mod.get_honcho_client(make_config(client_mod))
    honcho.get_metadata()
    srv.stop()  # route removed: listener closed and socket file unlinked
    for _ in range(3):
        with pytest.raises(HonchoConnectionError) as exc:
            honcho.get_metadata()
        assert _errno_names(exc.value) & {"ENOENT", "ECONNREFUSED"}
    assert canary.connections == 0


def test_timeout_is_bounded(client_mod, hermes_home, sock_dir):
    from honcho.http.exceptions import TimeoutError as HonchoTimeoutError
    srv = UdsServer(os.path.join(sock_dir, "h.sock"), hang=30)
    try:
        write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": srv.path,
                                        "timeout": 0.5})
        honcho = client_mod.get_honcho_client(make_config(client_mod))
        start = time.monotonic()
        with pytest.raises(HonchoTimeoutError):
            honcho.get_metadata()
        # SDK retries timeouts (DEFAULT_MAX_RETRIES); every attempt is capped at the configured 0.5 s.
        attempts = len(srv.seen)
        assert attempts >= 1
        assert time.monotonic() - start < attempts * 0.5 + 10
    finally:
        srv.server.hang = 0
        srv.stop()


# --- cache identity -------------------------------------------------------------------------------------

def test_explicit_cache_key_includes_socket_path(client_mod, hermes_home):
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/a.sock"})
    key_a = client_mod._client_cache_key(make_config(client_mod))
    key_b = client_mod._client_cache_key(make_config(client_mod, socket_path="/run/x/b.sock"))
    key_tcp = client_mod._client_cache_key(make_config(client_mod, socket_path=None))
    assert len({key_a, key_b, key_tcp}) == 3


def test_ambient_cache_key_follows_socket_path_changes(client_mod, cache_mod, hermes_home, monkeypatch):
    cache = cache_mod
    monkeypatch.setattr(client_mod, "resolve_active_host", lambda: "hermes")
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/a.sock"})
    key_a = cache._client_cache_key(None)
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/b.sock"})
    key_b = cache._client_cache_key(None)
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k"})
    key_tcp = cache._client_cache_key(None)
    assert key_a[-1] == repr("/run/x/a.sock") and key_b[-1] == repr("/run/x/b.sock") and key_tcp[-1] == ""
    assert cache._slot_identity(key_a) == cache._slot_identity(key_b)  # same identity: the old slot is evicted


def test_socket_path_change_rebuilds_and_isolates_clients(client_mod, hermes_home, sock_dir, canary):
    a = UdsServer(os.path.join(sock_dir, "a.sock"))
    b = UdsServer(os.path.join(sock_dir, "b.sock"))
    try:
        write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": a.path})
        cfg_a = make_config(client_mod)
        ha = client_mod.get_honcho_client(cfg_a)
        assert client_mod.get_honcho_client(cfg_a) is ha
        hb = client_mod.get_honcho_client(make_config(client_mod, socket_path=b.path))
        assert hb is not ha
        hb.get_metadata()
        assert b.seen and not a.seen and canary.connections == 0
        # Different profiles (hosts/paths) on different sockets stay separate slots.
        cfg_other = make_config(client_mod, host="hermes_other", hermes_home=hermes_home / "other")
        assert client_mod.get_honcho_client(cfg_other) is not hb
    finally:
        a.stop()
        b.stop()


# --- resource cleanup -----------------------------------------------------------------------------------

def test_close_honcho_clients_closes_injected_transport(client_mod, hermes_home, sock_dir):
    srv = UdsServer(os.path.join(sock_dir, "h.sock"))
    try:
        write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": srv.path})
        honcho = client_mod.get_honcho_client(make_config(client_mod))
        honcho.get_metadata()
        (http_client,) = list(client_mod._owned_http_clients.values())
        assert not http_client.is_closed
        client_mod.close_honcho_clients()
        assert http_client.is_closed and len(client_mod._owned_http_clients) == 0
    finally:
        srv.stop()


def test_failed_construction_closes_transport(client_mod, hermes_home, monkeypatch):
    captured = []

    def boom(**kw):
        captured.append(kw["http_client"])
        raise RuntimeError("sdk refused")

    monkeypatch.setattr("honcho.Honcho", boom)
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/h.sock"})
    with pytest.raises(RuntimeError):
        client_mod._build_client(make_config(client_mod))
    assert captured and captured[0].is_closed


def test_owned_transport_released_with_client(client_mod, hermes_home):
    write_honcho_json(hermes_home, {"baseUrl": "http://localhost", "apiKey": "k", "socketPath": "/run/x/h.sock"})
    honcho = client_mod._build_client(make_config(client_mod))
    (http_client,) = list(client_mod._owned_http_clients.values())
    del honcho
    gc.collect()
    assert len(client_mod._owned_http_clients) == 0 and http_client.is_closed


def test_plugin_never_uses_sdk_async_client(plugin):
    """The SDK's lazy ``.aio`` client is rebuilt from config WITHOUT ``http_client`` (it would dial TCP),
    so the plugin must stay on the sync client for the socket route to hold."""
    from pathlib import Path
    root = Path(plugin.__file__).parent
    offenders = [p.name for p in root.glob("*.py") if ".aio" in p.read_text(encoding="utf-8")]
    assert offenders == []
