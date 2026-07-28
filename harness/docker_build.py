"""Container construction following the SWE-bench remote-image lifecycle."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import docker

from swebenchpro.harness.docker_utils import cleanup_container, remove_image
from swebenchpro.harness.test_spec import TestSpec


class BuildImageError(Exception):
    def __init__(self, image_name: str, message: str, logger):
        super().__init__(message)
        self.image_name = image_name
        self.log_path = getattr(logger, "log_file", None)


def setup_logger(instance_id: str, log_file: Path, mode: str = "w", add_stdout: bool = False):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"swebenchpro.{instance_id}.{log_file.name}")
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if add_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.log_file = log_file
    return logger


def close_logger(logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def build_container(
    test_spec: TestSpec,
    client: docker.DockerClient,
    run_id: str,
    logger,
    nocache: bool = False,
    force_rebuild: bool = False,
    pull_image: bool = True,
):
    """Ensure the remote image exists and create, but do not start, its container."""

    image_name = test_spec.instance_image_key
    if force_rebuild:
        try:
            remove_image(client, image_name, logger)
        except Exception:
            pass

    try:
        client.images.get(image_name)
    except docker.errors.ImageNotFound:
        if not pull_image:
            raise BuildImageError(image_name, f"Image {image_name} is not available locally", logger)
        try:
            logger.info(f"Pulling image {image_name}")
            client.images.pull(image_name, platform=test_spec.platform)
        except Exception as exc:
            raise BuildImageError(image_name, str(exc), logger) from exc

    container = None
    try:
        create_kwargs = {
            "image": image_name,
            "name": test_spec.get_instance_container_name(run_id),
            "user": test_spec.user,
            "detach": True,
            "entrypoint": test_spec.entrypoint,
            "command": test_spec.command,
            "platform": test_spec.platform,
            "environment": test_spec.environment,
        }
        if test_spec.network_mode:
            create_kwargs["network_mode"] = test_spec.network_mode
        if test_spec.memory_limit:
            create_kwargs["mem_limit"] = test_spec.memory_limit
            create_kwargs["memswap_limit"] = test_spec.memory_limit
        container = client.containers.create(**create_kwargs)
        return container
    except Exception as exc:
        logger.error(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(exc), logger) from exc
