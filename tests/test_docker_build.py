from unittest.mock import Mock

import docker
import pytest

from swebenchpro.harness.docker_build import BuildImageError, build_container
from swebenchpro.harness.test_spec import TestSpec


def _spec():
    return TestSpec(
        instance_id="instance_owner__repo-1",
        image_name="example/sweap-images:owner.repo-instance",
        environment={"PATH": "/usr/bin"},
    )


def test_build_container_reuses_cached_image():
    client = Mock()
    container = Mock()
    client.containers.create.return_value = container

    result = build_container(_spec(), client, "run-1", Mock())

    assert result is container
    client.images.pull.assert_not_called()
    client.containers.create.assert_called_once()


def test_build_container_pulls_missing_image():
    client = Mock()
    container = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    client.containers.create.return_value = container

    result = build_container(_spec(), client, "run-1", Mock())

    assert result is container
    client.images.pull.assert_called_once_with(
        "example/sweap-images:owner.repo-instance",
        platform="linux/amd64",
    )


def test_build_container_requires_prefetched_image_when_pull_disabled():
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")

    with pytest.raises(BuildImageError):
        build_container(_spec(), client, "run-1", Mock(), pull_image=False)
