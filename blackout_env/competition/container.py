"""
Docker / CUDA MPS helpers for running each team's policy in its own container.

GPU split: on GPUs without MIG (e.g. RTX 3090) the only hard limit is CUDA MPS.
The MPS control daemon is started with server-wide defaults (SM percentage and pinned
memory per client). Client env vars can be changed from inside the container, so the
limits are set on the daemon, not only through client env vars.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

POLICY_IMAGE = "blackout-policy:latest"


@dataclass
class MpsDaemon:
    """CUDA MPS control daemon with a private pipe directory (does not touch /tmp/nvidia-mps)."""

    thread_percentage: int = 50
    mem_limit: str = "11G"
    device: int = 0
    root: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="blackout-mps-")))

    @property
    def pipe_dir(self) -> Path:
        return self.root / "pipe"

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(self.root / "log")
        env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        return env

    def _control(self, command: str) -> str:
        r = subprocess.run(
            ["nvidia-cuda-mps-control"], input=command + "\n", env=self._env(),
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()

    def start(self) -> None:
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "log").mkdir(parents=True, exist_ok=True)
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=self._env(), check=True)
        # Device index inside the daemon's view (CUDA_VISIBLE_DEVICES above) is 0.
        self._control(f"set_default_active_thread_percentage {self.thread_percentage}")
        self._control(f"set_default_device_pinned_mem_limit 0 {self.mem_limit}")

    def stop(self) -> None:
        self._control("quit")
        time.sleep(1.0)
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass
class TeamContainer:
    """
    One team's policy server: no network, read-only root, non-root user, CPU/RAM caps,
    GPU access only through the shared MPS daemon.
    """

    name: str
    submission_dir: Path
    mps: MpsDaemon | None
    image: str = POLICY_IMAGE
    cpus: float = 4.0
    memory: str = "16g"
    device: int = 0
    sock_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="blackout-sock-")))

    @property
    def socket_path(self) -> Path:
        return self.sock_dir / "policy.sock"

    def start(self) -> None:
        self.stop()
        os.chmod(self.sock_dir, 0o777)
        cmd = [
            "docker", "run", "-d",  # no --rm: keep logs if the server exits at load time
            "--name", self.name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=1g",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "512",
            "--cpus", str(self.cpus), "--memory", self.memory,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp",
            "-v", f"{self.submission_dir.resolve()}:/submission:ro",
            "-v", f"{self.sock_dir}:/sock",
        ]
        if self.mps is not None:
            cmd += [
                "--gpus", f"device={self.device}",
                "--ipc", "host",  # MPS clients share memory with the server through host IPC
                "-v", f"{self.mps.pipe_dir}:/mps",
                "-e", "CUDA_MPS_PIPE_DIRECTORY=/mps",
                "-e", f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={self.mps.thread_percentage}",
                "-e", f"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0={self.mps.mem_limit}",
            ]
        cmd.append(self.image)
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def wait_ready(self, timeout: float = 120.0) -> None:
        """Wait until the server opens its socket (model loaded and warmed up)."""
        deadline = time.monotonic() + timeout
        while not self.socket_path.exists():
            if not self.is_running():
                raise RuntimeError(f"{self.name} exited while loading the submission:\n{self.logs()}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.name} did not become ready in {timeout:.0f}s:\n{self.logs()}")
            time.sleep(0.2)

    def is_running(self) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.name],
            capture_output=True, text=True,
        )
        return r.stdout.strip() == "true"

    def logs(self, tail: int = 50) -> str:
        r = subprocess.run(["docker", "logs", "--tail", str(tail), self.name], capture_output=True, text=True)
        return r.stdout + r.stderr

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def cleanup(self) -> None:
        self.stop()
        shutil.rmtree(self.sock_dir, ignore_errors=True)
