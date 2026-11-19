from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _loopback_socket_ok() -> bool:
    """This CI sandbox intermittently forbids loopback sockets; TestClient and
    asyncio (self-pipe) both need them. Probe bind+listen plus three real
    connect attempts so affected tests skip honestly and run normally on a
    developer machine."""
    try:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        for _ in range(3):
            client = socket.socket()
            client.settimeout(2)
            client.connect(("127.0.0.1", port))
            client.close()
        listener.close()
        return True
    except OSError:
        return False


LOOPBACK_OK = _loopback_socket_ok()

requires_loopback = pytest.mark.skipif(
    not LOOPBACK_OK, reason="sandbox forbids loopback sockets (TestClient/asyncio)"
)


@pytest.fixture()
def ws(tmp_path) -> Path:
    return tmp_path / "workspace"