import socket
import pytest
from tests.conftest import OfflineNetworkError, _local_address


def test_public_network_is_rejected_before_io():
    for method in ("connect", "connect_ex"):
        with socket.socket() as sock, pytest.raises(OfflineNetworkError):
            getattr(sock, method)(("203.0.113.1", 443))
    with pytest.raises(OfflineNetworkError):
        socket.getaddrinfo("never-resolve.invalid", 443)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, pytest.raises(OfflineNetworkError):
        sock.sendto(b"blocked", ("203.0.113.1", 53))


def test_loopback_allowlist_is_exact():
    for host in ("127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"):
        assert _local_address((host, 80))
    for host in ("localhost.evil.example", "10.0.0.1", "0.0.0.0", "example.com"):
        assert not _local_address((host, 80))


def test_real_loopback_connection_remains_available():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0)); server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            connection, _ = server.accept(); connection.close()
