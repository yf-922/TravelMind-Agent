"""Explicit isolation for tests that construct offline Supervisor nodes."""

import os

import pytest


@pytest.fixture
def offline_supervisor_models(monkeypatch, tmp_path):
    from app.core import env
    from app.multi_agent_core import modification
    from app.planning import nodes

    # Reproduce a clean CI checkout, even on a developer machine with keys.
    monkeypatch.setattr(env, "LOCAL_ENV_FILE", tmp_path / "missing.env")
    for name in list(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")

    def fake_model(*args, **kwargs):
        return object()

    # Patch where each factory is looked up, including modification analysis.
    monkeypatch.setattr(nodes, "build_structured_llm", fake_model)
    monkeypatch.setattr(modification, "build_structured_llm", fake_model)


# Installed before test-module collection: SDK initialization is guarded too.
import ipaddress
import socket


class OfflineNetworkError(BaseException):
    """Not swallowed by provider fallback handlers catching Exception."""


def _local_address(address):
    if isinstance(address, str):  # AF_UNIX has no public network route.
        return True
    host = str(address[0]).lower()
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
        return ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def pytest_configure(config):
    if os.getenv("TRAVELMIND_TEST_ALLOW_NETWORK") == "1":
        if os.getenv("CI"):
            raise pytest.UsageError("CI cannot disable the offline network guard")
        return
    guard = pytest.MonkeyPatch()
    config._offline_network_guard = guard
    for method in ("connect", "connect_ex"):
        original = getattr(socket.socket, method)
        def guarded(sock, address, *args, _original=original, **kwargs):
            if not _local_address(address):
                raise OfflineNetworkError(f"Offline test blocked network connection: {address!r}")
            return _original(sock, address, *args, **kwargs)
        guard.setattr(socket.socket, method, guarded)
    original_resolve = socket.getaddrinfo
    def resolve(host, *args, **kwargs):
        # Refuse public DNS lookups before any resolver/network activity.
        if host is not None and not _local_address((host, 0)):
            raise OfflineNetworkError(f"Offline test blocked DNS resolution: {host!r}")
        return original_resolve(host, *args, **kwargs)
    guard.setattr(socket, "getaddrinfo", resolve)
    original_sendto = socket.socket.sendto
    def sendto(sock, data, *args):
        if args and not _local_address(args[-1]):
            raise OfflineNetworkError(f"Offline test blocked datagram: {args[-1]!r}")
        return original_sendto(sock, data, *args)
    guard.setattr(socket.socket, "sendto", sendto)


def pytest_unconfigure(config):
    guard = getattr(config, "_offline_network_guard", None)
    if guard:
        guard.undo()
