"""The script evaluates the performance of the SWEAP Pro agent using local Docker.

This evaluation script:
1. Takes a Hugging Face dataset and a JSON file containing patches
2. Pre-fetches all required Docker images serially
3. Runs each patch in a Docker container environment using Docker Hub images
4. Executes the tests using local run scripts and collects results
5. Calculates overall accuracy based on test pass/fail status

Usage:
python swe_bench_pro_eval.py \
    --dataset_name=ScaleAI/SWE-bench_Pro \
    --predictions_path={OUTPUT}/preds.json \
    --run_id=pro.test.01 \
    --report_dir=evaluation \
    --scripts_dir=run_scripts \
    --num_workers=5 \
    --dockerhub_username=jefzda \
    --use_local_docker 

The dataset must have columns: instance_id, before_repo_set_cmd, selected_test_files_to_run, 
  base_commit, base_dockerfile, instance_dockerfile, fail_to_pass, pass_to_pass

The patch file can be in JSON or JSONL format, and can be either:
- List format: [{"instance_id": "id", "patch": "content"}, ...]
- Dict format: {"id1": {"instance_id": "id1", "patch": "content"}, ...}

Field names supported: "patch" or "model_patch"
"""

import argparse
import concurrent.futures
import io
import json
import os
import platform as py_platform
import re
import tarfile
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

import docker
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from rich.console import Console

from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    remove_image,
)
from swebenchpro.helper_code.image_uri import get_dockerhub_image_uri


RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")

console = Console()


def cprint(message="", style=None, end="\n"):
    console.print(message, style=style, end=end, highlight=False)


def clog(message):
    console.log(message)


def cstream(message, end=""):
    console.print(message, end=end, markup=False, highlight=False, soft_wrap=True)


def vprint(verbose, message):
    if not verbose:
        return
    thread_name = threading.current_thread().name
    clog(f"[verbose][{thread_name}] {message}")


def normalize_model_name(model_name):
    # remove hosted/vllm/ from prefix
    if model_name and model_name.startswith("hosted_vllm/"):
        model_name = model_name[len("hosted_vllm/") :]
    return (model_name or "None").replace("/", "__")


def get_prediction_model_name(patch_sample):
    return (
        patch_sample.get("model_name_or_path")
        or patch_sample.get("model")
        or patch_sample.get("model_name")
        or "None"
    )


def get_instance_output_dir(output_dir, run_id, model_name, uid):
    return os.path.join(output_dir, run_id, normalize_model_name(model_name), uid)


def get_patch_status(patch_text):
    if not isinstance(patch_text, str) or not patch_text.strip():
        return "empty"
    if not patch_text.lstrip().startswith("diff --git"):
        return "error"
    return "ready"


def load_patches_from_file(patches_path):
    """
    Load patches/predictions from a file.
    Supports both JSON and JSONL formats, and both list and dict (keyed by instance_id) structures.
    
    Args:
        patches_path (str): Path to JSON or JSONL file containing patches
        
    Returns:
        list: List of patch samples, each with keys: instance_id, patch (or model_patch), optional prefix.
    """
    if patches_path.endswith(".jsonl"):
        with open(patches_path, "r") as f:
            patches = [json.loads(line) for line in f]
    else:  # .json
        with open(patches_path, "r") as f:
            data = json.load(f)
            # Handle both dict (keyed by instance_id) and list formats
            if isinstance(data, dict):
                patches = list(data.values())
            else:
                patches = data
    
    # Normalize field names: ensure each item has instance_id and patch field
    normalized_patches = []
    for patch_item in patches:
        if not isinstance(patch_item, dict):
            continue
        normalized = dict(patch_item)
        # Ensure instance_id is present
        if "instance_id" not in normalized:
            raise ValueError(f"Patch item missing instance_id: {patch_item}")
        # Normalize patch field name: use patch if present, else model_patch
        if "patch" not in normalized and "model_patch" in normalized:
            normalized["patch"] = normalized["model_patch"]
        # Also keep model_patch field for backward compatibility
        if "model_patch" not in normalized and "patch" in normalized:
            normalized["model_patch"] = normalized["patch"]

        normalized_patches.append(normalized)
    
    return normalized_patches


# Credit: prabhuteja12
def load_base_docker(iid):
    with open(f"swebenchpro/dockerfiles/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()

def instance_docker(iid):
    with open(f"swebenchpro/dockerfiles/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()

def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    with open(script_path, 'r') as f:
        return f.read()


def extract_dockerfile_env_vars(instance_id):
    """Extract ENV values from the base and instance Dockerfiles."""
    env_vars = {}
    for dockerfile_content in [
        load_base_docker(instance_id),
        instance_docker(instance_id),
    ]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if not line.startswith("ENV"):
                continue

            env_decl = line[3:].strip()
            key = value = None

            # Handle Dockerfile forms:
            # 1) ENV KEY=value (value may contain spaces/quotes)
            # 2) ENV KEY value
            key_value_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", env_decl)
            if key_value_match:
                key, value = key_value_match.group(1), key_value_match.group(2)
            else:
                env_parts = env_decl.split(None, 1)
                if len(env_parts) == 2:
                    key, value = env_parts
                else:
                    continue

            env_vars[key] = value.strip().strip("'").strip('"')

    return env_vars


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r'(?=^diff --git )', patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r'^Binary files .* differ$', section, re.MULTILINE):
            continue
        if re.search(r'^GIT binary patch$', section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)


def create_entryscript(sample):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]

    entry_script = f"""
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v --reject /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def create_dockerhub_tag(uid, repo_name=""):
    """
    Convert instance_id and repo name to Docker Hub compatible tag format.
    This must match the format used in the upload script.

    Args:
        uid (str): The instance_id (e.g., "django__django-12345")
        repo_name (str): The repository name from ECR (e.g., "sweap-images/nodebb.nodebb")

    Returns:
        str: Docker Hub compatible tag (e.g., "nodebb-nodebb-12345")
    """
    if repo_name:
        # For "NodeBB/NodeBB" -> repo_base="nodebb", repo_name="nodebb" 
        # Format: {repo_base}.{repo_name}-{OriginalCase}__{OriginalCase}-{hash}-{version}
        # Example: nodebb.nodebb-NodeBB__NodeBB-7b8bffd763e2155cf88f3ebc258fa68ebe18188d-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
        repo_base, repo_name_only = repo_name.lower().split("/")
        # Keep original case for the instance_id part (after removing "instance_" prefix)
        hsh = uid.replace("instance_", "")
        return f"{repo_base}.{repo_name_only}-{hsh}"
    else:
        image_name = "default"

    # Extract the tag part from the instance ID
    # For UIDs that start with a pattern like "django__django-", extract everything after position 9
    if "__" in uid and len(uid) > 9:
        tag_part = uid[9:]  # Skip the first 9 characters (e.g., "django__")
    else:
        tag_part = uid

    return f"{image_name}-{tag_part}"




def prepare_run(instance_dir, prefix, redo):
    os.makedirs(instance_dir, exist_ok=True)
    output_path = os.path.join(instance_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        cprint(f"Skipping {instance_dir} - output already exists", style="yellow")
        with open(output_path, "r") as f:
            return json.load(f), output_path, tempfile.mkdtemp()
    # Create workspace in a temporary directory instead of in logs
    workspace_dir = tempfile.mkdtemp(prefix="swe_bench_workspace_")
    return None, output_path, workspace_dir


def write_patch_snapshot(instance_dir, prefix, patch):
    with open(os.path.join(instance_dir, f"{prefix}_patch.diff"), "w") as f:
        f.write(patch)


def save_runscript_to_log(instance_dir, prefix, run_script):
    """Save the run script to the instance log directory for reference."""
    with open(os.path.join(instance_dir, f"{prefix}_run_script.sh"), "w") as f:
        f.write(run_script)


def assemble_workspace_files(uid, scripts_dir, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(sample)

    cleaned_patch = strip_binary_hunks(patch)
    if cleaned_patch != patch:
        cprint(f"Stripped binary diff hunks from patch for {uid}", style="yellow")

    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content




def write_files_to_container(container, files):
    """Write workspace files into /workspace using swebench harness copy helper."""
    container.exec_run("mkdir -p /workspace")
    with tempfile.TemporaryDirectory(prefix="swebenchpro_workspace_files_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for rel_path, content in files.items():
            local_file = tmp_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_text(content)
            copy_to_container(container, local_file, PurePosixPath(f"/workspace/{rel_path}"))


def _read_container_file(container, file_path):
    """Read a text file from container via get_archive; return None if missing."""
    try:
        stream, _ = container.get_archive(file_path)
    except Exception:
        return None

    tar_bytes = b"".join(stream)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = tar.getmembers()
        if not members:
            return None
        file_obj = tar.extractfile(members[0])
        if file_obj is None:
            return None
        return file_obj.read().decode("utf-8", errors="replace")


def save_entryscript_copy(instance_dir, prefix, entryscript_content):
    with open(os.path.join(instance_dir, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")



def collect_outputs_local_container(container, instance_dir, uid, prefix):
    stdout_content = _read_container_file(container, "/workspace/stdout.log") or ""
    with open(os.path.join(instance_dir, f"{prefix}_stdout.log"), "w") as f:
        f.write(stdout_content)

    stderr_content = _read_container_file(container, "/workspace/stderr.log") or ""
    with open(os.path.join(instance_dir, f"{prefix}_stderr.log"), "w") as f:
        f.write(stderr_content)

    output_json_text = _read_container_file(container, "/workspace/output.json")
    if output_json_text is None:
        cprint(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details",
            style="yellow",
        )
        return None

    try:
        output = json.loads(output_json_text)
    except json.JSONDecodeError as exc:
        cprint(f"Warning: Failed to parse output.json for {uid}: {exc}", style="yellow")
        return None

    with open(os.path.join(instance_dir, f"{prefix}_output.json"), "w") as f:
        json.dump(output, f)
    return output





def preflight_pull_images_serial(
    valid_patches,
    raw_sample_df,
    dockerhub_username,
    docker_platform=None,
    verbose=False,
    docker_api_timeout_seconds=120,
):
    """
    Pre-fetch all required Docker images serially before starting parallel execution.
    This avoids network contention and rate-limiting issues during threaded pulls.
    
    Args:
        valid_patches (list): List of (patch_sample, patch_text) tuples
        raw_sample_df (pd.DataFrame): Dataset rows indexed by instance_id
        dockerhub_username (str): Docker Hub username for images
        docker_platform (str): Optional platform override (e.g., linux/amd64)
        verbose (bool): If True, print per-image pull logs
    """
    unique_images = set()
    vprint(verbose, f"Starting preflight image discovery for {len(valid_patches)} runnable patches")
    for patch_sample, _ in valid_patches:
        uid = patch_sample["instance_id"]
        sample_repo = raw_sample_df.loc[uid].get("repo", "")
        image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample_repo)
        unique_images.add((image_uri, docker_platform))
    
    if not unique_images:
        vprint(verbose, "No images to prefetch; skipping preflight pull phase")
        return
    
    vprint(verbose, "Creating Docker client for preflight pulls")
    try:
        client = docker.from_env(timeout=int(docker_api_timeout_seconds))
    except Exception as e:
        cprint(f"Error creating Docker client for preflight pulls: {e}", style="red")
        return
    
    cprint(f"Pre-fetching {len(unique_images)} Docker images serially...", style="cyan")
    
    for idx, (image_uri, platform) in enumerate(unique_images, 1):
        try:
            if verbose:
                cprint(f"  [{idx}/{len(unique_images)}] Pulling {image_uri}...", end="")
            start_time = time.monotonic()
            if platform:
                client.images.pull(image_uri, platform=platform)
            else:
                client.images.pull(image_uri)
            if verbose:
                cprint(" OK", style="green")
            vprint(verbose, f"Prefetch success for {image_uri} in {time.monotonic() - start_time:.2f}s")
        except Exception as pull_err:
            # Try to check if image exists locally
            try:
                client.images.get(image_uri)
                if verbose:
                    cprint(" (already cached locally)", style="green")
                vprint(verbose, f"Prefetch pull failed but local cache exists for {image_uri}")
            except Exception:
                if verbose:
                    cprint(" FAILED", style="red")
                cprint(f"  Warning: Failed to pull image {image_uri}: {pull_err}", style="yellow")
                cprint("  Will attempt to use image during evaluation if available locally.", style="yellow")

    vprint(verbose, "Completed preflight image pull phase")


def eval_with_docker(
    patch,
    sample,
    output_dir,
    run_id,
    model_name,
    dockerhub_username,
    scripts_dir,
    prefix="",
    redo=False,
    block_network=False,
    docker_platform=None,
    remove_image_after_eval=False,
    skip_pull=False,
    verbose=False,
    entryscript_timeout_seconds=1800,
    docker_api_timeout_seconds=120,
):
    if docker is None:
        raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
    uid = sample["instance_id"]
    vprint(verbose, f"[{uid}] Starting eval_with_docker")
    instance_dir = get_instance_output_dir(output_dir, run_id, model_name, uid)
    existing_output, output_path, workspace_dir = prepare_run(instance_dir, prefix, redo)
    if existing_output is not None:
        vprint(verbose, f"[{uid}] Reusing existing output from {output_path}")
        return existing_output

    # print(f"Running local-docker evaluation for {uid}")

    container = None
    client = None
    dockerhub_image_uri = None
    try:
        try:
            vprint(verbose, f"[{uid}] Assembling workspace files from scripts dir: {scripts_dir}")
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            cprint(f"Error loading scripts for {uid}: {e}", style="red")
            return None
        vprint(verbose, f"[{uid}] Writing patch snapshot to {instance_dir}")
        write_patch_snapshot(instance_dir, prefix, patch)

        # Save run script to log directory for reference
        save_runscript_to_log(instance_dir, prefix, files["run_script.sh"])

        vprint(verbose, f"[{uid}] Extracting Docker ENV vars")
        env_vars = extract_dockerfile_env_vars(uid)
        vprint(verbose, f"[{uid}] Extracted {len(env_vars)} env vars")

        # Run container via Docker SDK
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        vprint(verbose, f"[{uid}] Resolved image URI: {dockerhub_image_uri}")

        vprint(verbose, f"[{uid}] Creating Docker client")
        client = docker.from_env(timeout=int(docker_api_timeout_seconds))
        if not skip_pull:
            try:
                pull_start = time.monotonic()
                vprint(verbose, f"[{uid}] Pulling image (skip_pull=False)")
                if docker_platform:
                    client.images.pull(dockerhub_image_uri, platform=docker_platform)
                else:
                    client.images.pull(dockerhub_image_uri)
                vprint(verbose, f"[{uid}] Pulled image in {time.monotonic() - pull_start:.2f}s")
            except Exception as pull_err:
                # If pull fails, fall back to a local image if present; otherwise, fail this run
                try:
                    client.images.get(dockerhub_image_uri)
                    cprint(f"Using locally available image: {dockerhub_image_uri}", style="yellow")
                    vprint(verbose, f"[{uid}] Pull failed; local image cache is available")
                except Exception:
                    cprint(f"Failed to pull or find image locally for {uid}: {pull_err}", style="red")
                    return None
        else:
            # Image should already be cached from preflight phase
            try:
                vprint(verbose, f"[{uid}] Verifying pre-fetched image exists locally")
                client.images.get(dockerhub_image_uri)
                vprint(verbose, f"[{uid}] Image exists locally")
            except Exception:
                cprint(f"Warning: Image {dockerhub_image_uri} not found locally for {uid}", style="yellow")
                return None

        run_kwargs = {
            "detach": True,
            "remove": False,
            "entrypoint": "/bin/bash",  # Override image entrypoint
            "command": ["-lc", "tail -f /dev/null"],
            "environment": env_vars,
        }
        if block_network:
            run_kwargs["network_mode"] = "none"
        # Optional platform override (useful on Apple Silicon)
        if docker_platform:
            run_kwargs["platform"] = docker_platform

        vprint(verbose, f"[{uid}] Starting container")
        container_start = time.monotonic()
        container = client.containers.run(dockerhub_image_uri, **run_kwargs)
        vprint(verbose, f"[{uid}] Container started in {time.monotonic() - container_start:.2f}s")

        vprint(verbose, f"[{uid}] Uploading workspace files to container")
        write_files_to_container(container, files)
        vprint(verbose, f"[{uid}] Workspace upload complete")
        
        entryscript_cmd = "bash /workspace/entryscript.sh"
        vprint(
            verbose,
            f"[{uid}] Running entryscript via harness exec_run_with_timeout (timeout={int(entryscript_timeout_seconds)}s)",
        )
        test_output, timed_out, total_runtime = exec_run_with_timeout(
            container,
            entryscript_cmd,
            timeout=int(entryscript_timeout_seconds),
        )
        vprint(verbose, f"[{uid}] Entryscript runtime: {total_runtime:.2f}s")
        if verbose and test_output:
            cstream(test_output)

        if timed_out:
            cprint(
                f"Entryscript timed out for {uid} after {int(entryscript_timeout_seconds)}s",
                style="red",
            )
            return None
        # Collect outputs and logs, and save entryscript for reference
        vprint(verbose, f"[{uid}] Collecting output artifacts")
        output = collect_outputs_local_container(container, instance_dir, uid, prefix)
        if output is None:
            vprint(verbose, f"[{uid}] Output collection returned None")
            return None
        save_entryscript_copy(instance_dir, prefix, entryscript_content)
        vprint(verbose, f"[{uid}] Evaluation completed successfully")

        return output
    except Exception as e:
        cprint(f"Error in eval_with_docker for {uid}: {repr(e)}", style="red")
        cprint(f"Error type: {type(e)}", style="red")
        vprint(verbose, f"[{uid}] Exception path taken")
        return None
    finally:
        vprint(verbose, f"[{uid}] Entering cleanup")
        if container is not None:
            try:
                if client is None:
                    client = docker.from_env()
                cleanup_container(client, container, logger="quiet")
                vprint(verbose, f"[{uid}] Container removed")
            except Exception:
                vprint(verbose, f"[{uid}] Container removal failed")
                pass
        if remove_image_after_eval and dockerhub_image_uri:
            try:
                client = docker.from_env(timeout=int(docker_api_timeout_seconds))
                remove_image(client, dockerhub_image_uri, logger="quiet")
                cprint(f"Removed Docker image: {dockerhub_image_uri}", style="yellow")
                vprint(verbose, f"[{uid}] Image removed during cleanup")
            except Exception as e:
                cprint(f"Warning: failed to remove Docker image {dockerhub_image_uri}: {e}", style="yellow")
        vprint(verbose, f"[{uid}] Cleanup complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWEAP Pro evaluations using local Docker with Docker Hub images and local scripts")
    parser.add_argument(
        "-id",
        "--run_id",
        required=True,
        help="Run ID - identifies the run",
    )
    parser.add_argument("--dataset_name", required=True, help="Hugging Face dataset name or path (e.g., 'princeton-nlp/SWE-Bench')")
    parser.add_argument(
        "--predictions_path", required=True, help="Path to the predictions file (JSON or JSONL)"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="evaluation",
        help="Directory to write finalized run reports",
    )
    parser.add_argument(
        "--dockerhub_username", help="Docker Hub username where sweap-images repository is located", default="jefzda"
    )
    parser.add_argument(
        "--scripts_dir", help="Directory containing local run scripts (e.g., scripts/run_scripts)", default="swebenchpro/run_scripts"
    )
    parser.add_argument(
        "--docker_platform",
        default=None,
        help="Docker platform override, e.g., linux/amd64; defaults to auto-detect",
    )
    parser.add_argument(
        "--redo", action="store_true", help="Redo evaluations even if output exists"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to run evaluations in parallel",
    )
    parser.add_argument(
        "--block_network", action="store_true", help="Block network access inside container"
    )
    parser.add_argument(
        "--remove_image_after_eval",
        action="store_true",
        help="Remove pulled Docker image after each local Docker evaluation to save disk space",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (per-image pull logs and streamed container output)",
    )
    parser.add_argument(
        "--heartbeat_seconds",
        type=float,
        default=30.0,
        help="Heartbeat interval in seconds while waiting for worker completion",
    )
    parser.add_argument(
        "--entryscript_timeout_seconds",
        type=int,
        default=1800,
        help="Timeout in seconds for /workspace/entryscript.sh inside the container",
    )
    parser.add_argument(
        "--docker_api_timeout_seconds",
        type=int,
        default=120,
        help="Timeout in seconds for Docker SDK API calls (pull/get/run/remove)",
    )
    return parser.parse_args()


def build_run_style_report(raw_sample_df, patches_to_run, patch_statuses, eval_results):
    """Build a summary report similar to swebench.harness.reporting.make_run_report."""
    total_ids = set(raw_sample_df.index.tolist())
    submitted_ids = {p.get("instance_id") for p in patches_to_run if isinstance(p, dict) and p.get("instance_id")}

    completed_ids = {iid for iid, status in patch_statuses.items() if status in {"pass", "fail"}}
    resolved_ids = {iid for iid, status in patch_statuses.items() if status == "pass"}
    unresolved_ids = {iid for iid, status in patch_statuses.items() if status == "fail"}
    empty_patch_ids = {iid for iid, status in patch_statuses.items() if status == "empty"}
    error_ids = {iid for iid, status in patch_statuses.items() if status == "error"}
    incomplete_ids = total_ids - submitted_ids

    return {
        "total_instances": len(total_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "submitted_ids": sorted(submitted_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
        "schema_version": 2,
        # Keep per-instance booleans available for downstream consumers.
        # "instance_results": eval_results,
    }


def main():
    args = parse_args()
    run_start = time.monotonic()
    vprint(args.verbose, "Starting run_evaluation main()")
    if not args.run_id:
        raise ValueError("Run ID must be provided")

    output_root = RUN_EVALUATION_LOG_DIR
    run_output_dir = output_root / args.run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    vprint(args.verbose, f"Run output dir: {run_output_dir}")
    vprint(args.verbose, f"Report dir: {report_dir}")

    # Load dataset from Hugging Face
    cprint(f"Loading dataset: {args.dataset_name}", style="cyan")
    dataset_load_start = time.monotonic()
    dataset = load_dataset(args.dataset_name)
    vprint(args.verbose, f"Dataset loaded in {time.monotonic() - dataset_load_start:.2f}s")
    
    # Handle different dataset splits (use 'test' if available, else 'train' or first split)
    if isinstance(dataset, dict):   
        if 'test' in dataset:
            dataset = dataset['test']
        else:
            # Get the first available split
            dataset = dataset[list(dataset.keys())[0]]
    
    # Convert to DataFrame
    vprint(args.verbose, "Converting dataset to pandas DataFrame")
    raw_sample_df = dataset.to_pandas()
    
    # Replace nulls with empty strings
    raw_sample_df = raw_sample_df.fillna("")
    
    # Ensure instance_id column exists and use it as index
    if 'instance_id' not in raw_sample_df.columns:
        raise ValueError("Dataset must contain 'instance_id' column")
    raw_sample_df = raw_sample_df.set_index("instance_id", drop=False)
    vprint(args.verbose, f"Dataset rows available: {len(raw_sample_df)}")

    # Load patches from file (supports JSON/JSONL, dict/list formats)
    vprint(args.verbose, f"Loading patches from {args.predictions_path}")
    patches_to_run = load_patches_from_file(args.predictions_path)
    vprint(args.verbose, f"Loaded {len(patches_to_run)} patches")
    
    eval_results = {}
    patch_statuses = {}
    status_counts = {"pass": 0, "fail": 0, "error": 0, "empty": 0}
    report_sample = patches_to_run[0] if patches_to_run else {}
    report_model_name = normalize_model_name(get_prediction_model_name(report_sample))

    # Filter patches to only include those with matching instance_ids in the raw sample data
    valid_patches = []
    missing_instances = []
    for patch_sample in patches_to_run:
        instance_id = patch_sample["instance_id"]
        patch_text = patch_sample.get("model_patch", patch_sample.get("patch", ""))

        if instance_id not in raw_sample_df.index:
            missing_instances.append(instance_id)
            patch_statuses[instance_id] = "error"
            status_counts["error"] += 1
            eval_results[instance_id] = False
            continue

        patch_status = get_patch_status(patch_text)
        if patch_status != "ready":
            patch_statuses[instance_id] = patch_status
            status_counts[patch_status] += 1
            eval_results[instance_id] = False
            continue

        valid_patches.append((patch_sample, patch_text))
        # print(f"Found patch for instance_id: {instance_id}")

    vprint(
        args.verbose,
        f"Patch filtering complete: runnable={len(valid_patches)}, missing={len(missing_instances)}, empty={status_counts['empty']}, pre-error={status_counts['error']}",
    )
    
    if missing_instances:
        cprint(f"Warning: Found {len(missing_instances)} patch instances not in raw sample data:", style="yellow")
        for missing_id in missing_instances[:5]:  # Show first 5
            cprint(f"  - {missing_id}")
        if len(missing_instances) > 5:
            cprint(f"  ... and {len(missing_instances) - 5} more")

    if status_counts["empty"] or status_counts["error"]:
        cprint(
            f"Skipping {status_counts['empty']} empty patches and {status_counts['error']} errored patches before execution",
            style="yellow",
        )

    cprint(
        f"Proceeding with {len(valid_patches)} runnable patches out of {len(patches_to_run)} total patches",
        style="cyan",
    )

    # Auto-detect default platform if not provided: prefer linux/amd64 on Apple Silicon
    detected_platform = None
    if args.docker_platform is None:
        try:
            if py_platform.machine().lower() in {"arm64", "aarch64"}:
                detected_platform = "linux/amd64"
        except Exception:
            detected_platform = None

    # Pre-fetch all Docker images serially
    preflight_start = time.monotonic()
    preflight_pull_images_serial(
        valid_patches,
        raw_sample_df,
        args.dockerhub_username,
        args.docker_platform or detected_platform,
        verbose=args.verbose,
        docker_api_timeout_seconds=args.docker_api_timeout_seconds,
    )
    vprint(args.verbose, f"Preflight phase finished in {time.monotonic() - preflight_start:.2f}s")

    # Run evaluations (serial if num_workers <= 1; threaded otherwise)
    stats_lock = threading.Lock() if args.num_workers > 1 else None
    pbar = tqdm(total=len(valid_patches), desc="Evaluation", postfix=status_counts)

    def run_evaluation_with_progress(patch_sample, patch_text):
        instance_id = patch_sample["instance_id"]
        patch_status = "error"
        instance_start = time.monotonic()
        vprint(args.verbose, f"[{instance_id}] Worker started")

        try:
            output = eval_with_docker(
                patch_text,
                raw_sample_df.loc[instance_id],
                str(output_root),
                args.run_id,
                get_prediction_model_name(patch_sample),
                args.dockerhub_username,
                args.scripts_dir,
                prefix=patch_sample.get("prefix", ""),
                redo=args.redo,
                block_network=args.block_network,
                docker_platform=args.docker_platform or detected_platform,
                remove_image_after_eval=args.remove_image_after_eval,
                skip_pull=True,  # Images already pre-fetched
                verbose=args.verbose,
                entryscript_timeout_seconds=args.entryscript_timeout_seconds,
                docker_api_timeout_seconds=args.docker_api_timeout_seconds,
            )

            if output is None:
                cprint(f"Evaluation for {instance_id} returned None", style="yellow")
            else:
                raw_sample = raw_sample_df.loc[instance_id]
                passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
                f2p = set(eval(raw_sample["fail_to_pass"]))
                p2p = set(eval(raw_sample["pass_to_pass"]))
                result = (f2p | p2p) <= passed_tests
                patch_status = "pass" if result else "fail"
                eval_results[instance_id] = result
        except Exception as exc:
            cprint(f"Evaluation for {instance_id} generated an exception: {exc}", style="red")
            patch_status = "error"

        if stats_lock is not None:
            with stats_lock:
                if patch_status == "error":
                    eval_results[instance_id] = False
                patch_statuses[instance_id] = patch_status
                status_counts[patch_status] += 1

                current_accuracy = status_counts["pass"] / max(1, sum(status_counts.values()))
                pbar.set_description(f"Accuracy: {current_accuracy:.2%}")
                pbar.set_postfix(status_counts)
                pbar.update(1)
        else:
            if patch_status == "error":
                eval_results[instance_id] = False
            patch_statuses[instance_id] = patch_status
            status_counts[patch_status] += 1

            current_accuracy = status_counts["pass"] / max(1, sum(status_counts.values()))
            pbar.set_description(f"Accuracy: {current_accuracy:.2%}")
            pbar.set_postfix(status_counts)
            pbar.update(1)
        vprint(
            args.verbose,
            f"[{instance_id}] Worker finished with status={patch_status} in {time.monotonic() - instance_start:.2f}s",
        )

    if args.num_workers <= 1:
        vprint(args.verbose, "Running in serial mode (num_workers <= 1); no thread pool will be used")
        for patch_sample, patch_text in valid_patches:
            run_evaluation_with_progress(patch_sample, patch_text)
    else:
        heartbeat_seconds = max(1.0, args.heartbeat_seconds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            vprint(args.verbose, f"Submitting {len(valid_patches)} tasks to ThreadPoolExecutor(max_workers={args.num_workers})")
            futures = [
                executor.submit(run_evaluation_with_progress, patch_sample, patch_text)
                for patch_sample, patch_text in valid_patches
            ]
            pending = set(futures)
            wait_start = time.monotonic()
            cprint(f"Heartbeat enabled: every {heartbeat_seconds:.1f}s while waiting for workers", style="cyan")
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=heartbeat_seconds,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    vprint(args.verbose, "A worker future completed")
                    future.result()

                if pending and not done:
                    elapsed = time.monotonic() - wait_start
                    completed = len(futures) - len(pending)
                    cprint(
                        f"Heartbeat: waiting on {len(pending)} workers "
                        f"({completed}/{len(futures)} done, elapsed {elapsed:.1f}s)",
                        style="magenta",
                    )

    pbar.close()
    report = build_run_style_report(raw_sample_df, patches_to_run, patch_statuses, eval_results)
    report_path = report_dir / f"{report_model_name}.{args.run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    overall_accuracy = (status_counts["pass"] / len(eval_results)) if eval_results else 0.0
    cprint(f"Overall accuracy: {overall_accuracy:.4f}", style="bold green")
    cprint(
        "Summary: \n"
        f"pass={status_counts['pass']}, \n"
        f"fail={status_counts['fail']}, \n"
        f"error={status_counts['error']}, \n"
        f"empty={status_counts['empty']}"
    )
    cprint(f"Report written to {report_path}", style="green")
    vprint(args.verbose, f"Run finished in {time.monotonic() - run_start:.2f}s")


if __name__ == "__main__":
    main()
