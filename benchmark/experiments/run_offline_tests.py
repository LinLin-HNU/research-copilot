"""Run unittest discovery with non-loopback socket connections disabled.

Fake FastAPI TestClient transports do not need sockets. A forgotten mock may
not silently invoke a real model, even when a developer's .env is present.
"""
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def deny_network(*args, **kwargs):
    raise AssertionError("Offline test attempted a network connection")


def local_only(original):
    def connect(sock, address):
        # Windows asyncio creates a loopback socket pair for its internal pipe.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original(sock, address)
        if sock.family == getattr(socket, "AF_UNIX", None):
            return original(sock, address)
        return deny_network()
    return connect


def main():
    with patch.object(socket.socket, "connect", local_only(socket.socket.connect)), \
         patch.object(socket.socket, "connect_ex", local_only(socket.socket.connect_ex)), \
         patch.object(socket, "create_connection", deny_network):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
