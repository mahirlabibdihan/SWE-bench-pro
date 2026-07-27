"""Docker helpers mirrored from the SWE-bench harness."""

from __future__ import annotations

import tarfile
import threading
import time
from pathlib import Path

import docker
from docker.models.containers import Container


def copy_to_container(container: Container, src: Path, dst: Path) -> None:
    if not dst.parent:
        raise ValueError(f"Destination path parent directory cannot be empty: {dst}")

    tar_path = src.with_suffix(src.suffix + ".tar")
    try:
        with tarfile.open(tar_path, "w") as tar:
            tar.add(src, arcname=dst.name)
        container.exec_run(f"mkdir -p {dst.parent}")
        container.put_archive(str(dst.parent), tar_path.read_bytes())
    finally:
        tar_path.unlink(missing_ok=True)


def remove_image(client, image_id: str, logger=None) -> None:
    try:
        client.images.remove(image_id, force=True)
    except docker.errors.ImageNotFound:
        return


def cleanup_container(client, container, logger=None) -> None:
    if not container:
        return
    try:
        container.stop(timeout=15)
    except Exception:
        try:
            container.kill()
        except Exception:
            pass
    container.remove(force=True)


def exec_run_with_timeout(container, cmd: str, timeout: int | None = 60):
    """Run a command using the same return contract as SWE-bench."""

    exec_result = bytearray()
    exec_id = None
    exception = None

    def run_command() -> None:
        nonlocal exec_id, exception
        try:
            exec_id = container.client.api.exec_create(container.id, cmd)["Id"]
            for chunk in container.client.api.exec_start(exec_id, stream=True):
                exec_result.extend(chunk)
        except Exception as exc:
            exception = exc

    thread = threading.Thread(target=run_command, daemon=True)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if exception is not None:
        raise exception

    timed_out = thread.is_alive()
    if timed_out and exec_id is not None:
        try:
            exec_pid = container.client.api.exec_inspect(exec_id)["Pid"]
            container.exec_run(f"kill -TERM {exec_pid}", detach=True)
        except Exception:
            pass

    return exec_result.decode(errors="replace"), timed_out, time.time() - start_time
