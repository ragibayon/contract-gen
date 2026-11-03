from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        for target in [
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
            "socket.getaddrinfo",
        ]:
            guard = patch(
                target,
                side_effect=AssertionError(
                    "Readiness tests must not access the network"
                ),
            )
            guard.start()
            self.addCleanup(guard.stop)


def isolated_python(script: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    try:
        import resource
    except ImportError:
        raise unittest.SkipTest(
            "Bounded program execution requires POSIX resource limits"
        ) from None

    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))

    prefix = f"import sys\nsys.path.insert(0, {str(ROOT / 'src')!r})\nsys.path.insert(0, {str(ROOT)!r})\n"
    prefix += "from unittest.mock import patch\n_network_guards = [patch(target, side_effect=AssertionError('Offline replay attempted network access')) for target in ['socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection', 'socket.getaddrinfo']]\nfor guard in _network_guards: guard.start()\n"
    with tempfile.TemporaryDirectory(prefix="contract-readiness-") as tmp:
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", prefix + script],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=set_limits,
        )
