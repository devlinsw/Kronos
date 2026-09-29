import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run-tailnet.sh"
FAKE_BIN = Path(__file__).resolve().parent / "fakes"


def make_fake_env(tmp_path: Path, tailscale_ip: str) -> tuple[dict[str, str], Path]:
    capture = tmp_path / "docker-capture.txt"
    env = os.environ.copy()
    env.pop("KRONOS_TAILSCALE_IP", None)
    env.update({
        "PATH": f"{FAKE_BIN}{os.pathsep}{env['PATH']}",
        "FAKE_TAILSCALE_IP": tailscale_ip,
        "TEST_CAPTURE": str(capture),
    })
    return env, capture


def test_launcher_uses_verified_host_tailscale_address(tmp_path):
    env, capture = make_fake_env(tmp_path, "100.64.0.42")
    result = subprocess.run(
        [str(LAUNCHER), "config", "--services"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ["100.64.0.42", "compose config --services"]


def test_launcher_rejects_configured_address_that_differs_from_tailscale(tmp_path):
    env, capture = make_fake_env(tmp_path, "100.64.0.42")
    env["KRONOS_TAILSCALE_IP"] = "100.64.0.99"
    result = subprocess.run(
        [str(LAUNCHER), "config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not capture.exists()


def test_launcher_rejects_non_tailscale_interface_address(tmp_path):
    env, capture = make_fake_env(tmp_path, "192.0.2.42")
    result = subprocess.run(
        [str(LAUNCHER), "config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "valid Tailscale IPv4" in result.stderr
    assert not capture.exists()
