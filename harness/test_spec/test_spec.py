from dataclasses import dataclass, field


@dataclass
class TestSpec:
    """Container specification for one prebuilt SWE-bench Pro image."""

    instance_id: str
    image_name: str
    platform: str = "linux/amd64"
    user: str = "root"
    entrypoint: str | None = "/bin/bash"
    command: list[str] | str = field(default_factory=lambda: ["-lc", "tail -f /dev/null"])
    environment: dict[str, str] = field(default_factory=dict)
    network_mode: str | None = None
    memory_limit: str | int | None = None

    @property
    def instance_image_key(self) -> str:
        return self.image_name

    @property
    def is_remote_image(self) -> bool:
        return True

    def get_instance_container_name(self, run_id: str | None = None) -> str:
        base = f"sweb.eval.{self.instance_id.lower()}"
        return f"{base}.{run_id}" if run_id else base
