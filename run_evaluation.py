"""
The script is used to evaluate the performance of the SWEAP Pro agent with Modal.

This evaluation script:
1. Takes a Hugging Face dataset and a JSON file containing patches
2. Runs each patch in a Modal sandbox environment using Docker Hub images
3. Executes the tests using local run scripts and collects results
4. Calculates overall accuracy based on test pass/fail status

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
from pathlib import Path

try:
    import modal  # Lazy/optional: only required when not using --use_local_docker
except Exception:
    modal = None
try:
    import docker  # Optional: used when --use_local_docker is set
except Exception:
    docker = None
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from swebenchpro.helper_code.image_uri import get_dockerhub_image_uri


RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")


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
git apply -v /workspace/patch.diff
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
        print(f"Skipping {instance_dir} - output already exists")
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
        print(f"Stripped binary diff hunks from patch for {uid}")

    files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def write_files_modal(sandbox, files):
    for rel_path, content in files.items():
        with sandbox.open(f"/workspace/{rel_path}", "w") as f:
            f.write(content)


def write_files_local(workspace_dir, files):
    for rel_path, content in files.items():
        dst = os.path.join(workspace_dir, rel_path)
        with open(dst, "w") as f:
            f.write(content)


def _create_workspace_tar(files):
    """Create a tar archive (bytes) for workspace files to upload into a container."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel_path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=rel_path)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf.read()


def write_files_to_container(container, files):
    """Write workspace files directly into /workspace inside a running container."""
    container.exec_run("mkdir -p /workspace")
    tar_data = _create_workspace_tar(files)
    container.put_archive("/workspace", tar_data)


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


def exec_and_stream_container_command(container, cmd, uid=""):
    """Execute a command in a running container and stream stdout/stderr to host console."""
    exec_id = container.client.api.exec_create(container.id, cmd)["Id"]
    prefix = f"[{uid}] " if uid else ""
    print(f"{prefix}Running: {cmd}")

    stream = container.client.api.exec_start(exec_id, stream=True, demux=True)
    for stdout_chunk, stderr_chunk in stream:
        if stdout_chunk:
            print(f"{prefix}{stdout_chunk.decode('utf-8', errors='replace')}", end="")
        if stderr_chunk:
            print(f"{prefix}{stderr_chunk.decode('utf-8', errors='replace')}", end="")

    inspect_data = container.client.api.exec_inspect(exec_id)
    return inspect_data.get("ExitCode", 1)


def save_entryscript_copy(instance_dir, prefix, entryscript_content):
    with open(os.path.join(instance_dir, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def collect_outputs_modal(sandbox, instance_dir, uid, prefix):
    # Save logs first (best-effort)
    try:
        with sandbox.open("/workspace/stdout.log", "r") as f_in:
            with open(os.path.join(instance_dir, f"{prefix}_stdout.log"), "w") as f:
                stdout_content = f_in.read()
                f.write(stdout_content if stdout_content is not None else "")
    except FileNotFoundError:
        pass
    try:
        with sandbox.open("/workspace/stderr.log", "r") as f_in:
            with open(os.path.join(instance_dir, f"{prefix}_stderr.log"), "w") as f:
                stderr_content = f_in.read()
                f.write(stderr_content if stderr_content is not None else "")
    except FileNotFoundError:
        pass

    # Then try to read output.json
    try:
        with sandbox.open("/workspace/output.json", "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(instance_dir, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def collect_outputs_local(workspace_dir, instance_dir, uid, prefix):
    def _copy_safe(src_name, dest_name):
        src_path = os.path.join(workspace_dir, src_name)
        dest_path = os.path.join(instance_dir, dest_name)
        try:
            with open(src_path, "r") as f_in:
                content = f_in.read()
        except FileNotFoundError:
            content = ""
        with open(dest_path, "w") as f_out:
            f_out.write(content if content is not None else "")

    _copy_safe("stdout.log", f"{prefix}_stdout.log")
    _copy_safe("stderr.log", f"{prefix}_stderr.log")

    # Then try to read output.json
    try:
        with open(os.path.join(workspace_dir, "output.json"), "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(instance_dir, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def collect_outputs_local_container(container, instance_dir, uid, prefix):
    stdout_content = _read_container_file(container, "/workspace/stdout.log") or ""
    with open(os.path.join(instance_dir, f"{prefix}_stdout.log"), "w") as f:
        f.write(stdout_content)

    stderr_content = _read_container_file(container, "/workspace/stderr.log") or ""
    with open(os.path.join(instance_dir, f"{prefix}_stderr.log"), "w") as f:
        f.write(stderr_content)

    output_json_text = _read_container_file(container, "/workspace/output.json")
    if output_json_text is None:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None

    try:
        output = json.loads(output_json_text)
    except json.JSONDecodeError as exc:
        print(f"Warning: Failed to parse output.json for {uid}: {exc}")
        return None

    with open(os.path.join(instance_dir, f"{prefix}_output.json"), "w") as f:
        json.dump(output, f)
    return output


def eval_with_modal(patch, sample, output_dir, run_id, model_name, dockerhub_username, scripts_dir, prefix="", redo=False, block_network=False, docker_platform=None):
    if modal is None:
        raise RuntimeError("modal is not installed. Install it or run with --use_local_docker")
    uid = sample["instance_id"]
    instance_dir = get_instance_output_dir(output_dir, run_id, model_name, uid)
    existing_output, output_path, workspace_dir = prepare_run(instance_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    sandbox = None
    
    print(f"Running evaluation for {uid}")
    try:
        write_patch_snapshot(instance_dir, prefix, patch)

        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None

        # Save run script to log directory for reference
        save_runscript_to_log(instance_dir, prefix, files["run_script.sh"])

        app = modal.App.lookup(name="swe-bench-pro-eval", create_if_missing=True)
        
        # Use Docker Hub image instead of ECR
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        # print(f"Using Docker Hub image: {dockerhub_image_uri}")
        
        image = modal.Image.from_registry(
            dockerhub_image_uri
        )

        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=60 * 60,
            cpu=(1, 4),
            memory=(5 * 1024, 30 * 1024),
            block_network=block_network,
        )
        
        process = sandbox.exec("mkdir", "-p", "/workspace")
        process.wait()
        
        write_files_modal(sandbox, files)
            
        process = sandbox.exec("bash", "/workspace/entryscript.sh")
        process.wait()
        
        # Check if the process was successful
        if process.returncode != 0:
            print(f"Entryscript failed for {uid} with return code: {process.returncode}")
            # Get stderr from the process directly (note: this may not work with all Modal versions)
            try:
                stderr_content = getattr(process, 'stderr', None)
                if stderr_content and hasattr(stderr_content, 'read'):
                    error_details = stderr_content.read()
                    if error_details:
                        print(f"Error details for {uid}:")
                        print(error_details[:1000])  # Print first 1000 chars
            except Exception as e:
                print(f"Failed to read stderr for {uid}: {e}")
            
        output = collect_outputs_modal(sandbox, instance_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(instance_dir, prefix, entryscript_content)
            
        return output
    except Exception as e:
        print(f"Error in eval_with_modal for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None
    finally:
        if sandbox:
            try:
                sandbox.terminate()
            except Exception:
                pass


def eval_with_docker(patch, sample, output_dir, run_id, model_name, dockerhub_username, scripts_dir, prefix="", redo=False, block_network=False, docker_platform=None):
    if docker is None:
        raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
    uid = sample["instance_id"]
    instance_dir = get_instance_output_dir(output_dir, run_id, model_name, uid)
    existing_output, output_path, workspace_dir = prepare_run(instance_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    # print(f"Running local-docker evaluation for {uid}")

    container = None
    try:
        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_patch_snapshot(instance_dir, prefix, patch)

        # Save run script to log directory for reference
        save_runscript_to_log(instance_dir, prefix, files["run_script.sh"])

        env_vars = extract_dockerfile_env_vars(uid)

        # Run container via Docker SDK
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        # print(f"Using Docker Hub image: {dockerhub_image_uri}")

        client = docker.from_env()
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            # If pull fails, fall back to a local image if present; otherwise, fail this run
            try:
                client.images.get(dockerhub_image_uri)
                print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
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

        container = client.containers.run(dockerhub_image_uri, **run_kwargs)

        write_files_to_container(container, files)

        # status_code = exec_and_stream_container_command(
        #     container,
        #     "bash /workspace/entryscript.sh",
        #     uid=uid,
        # )
        
        exec_result = container.exec_run("bash /workspace/entryscript.sh")
        status_code = exec_result.exit_code

        if status_code != 0:
            print(f"Entryscript failed for {uid} with return code: {status_code}")
        # Collect outputs and logs, and save entryscript for reference
        output = collect_outputs_local_container(container, instance_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(instance_dir, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWEAP Pro evaluations using Modal or local Docker with Docker Hub images and local scripts")
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
        default=".",
        help="Directory to write finalized run reports",
    )
    parser.add_argument(
        "--dockerhub_username", required=True, help="Docker Hub username where sweap-images repository is located"
    )
    parser.add_argument(
        "--scripts_dir", required=True, help="Directory containing local run scripts (e.g., scripts/run_scripts)"
    )
    parser.add_argument(
        "--use_local_docker", action="store_true", help="Run locally with Docker instead of Modal"
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
        default=50,
        help="Number of workers to run evaluations in parallel",
    )
    parser.add_argument(
        "--block_network", action="store_true", help="Block network access inside container"
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
        "instance_results": eval_results,
    }


def main():
    args = parse_args()
    if not args.run_id:
        raise ValueError("Run ID must be provided")

    output_root = RUN_EVALUATION_LOG_DIR
    run_output_dir = output_root / args.run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from Hugging Face
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    # Handle different dataset splits (use 'test' if available, else 'train' or first split)
    if isinstance(dataset, dict):   
        if 'test' in dataset:
            dataset = dataset['test']
        else:
            # Get the first available split
            dataset = dataset[list(dataset.keys())[0]]
    
    # Convert to DataFrame
    raw_sample_df = dataset.to_pandas()
    
    # Replace nulls with empty strings
    raw_sample_df = raw_sample_df.fillna("")
    
    # Ensure instance_id column exists and use it as index
    if 'instance_id' not in raw_sample_df.columns:
        raise ValueError("Dataset must contain 'instance_id' column")
    raw_sample_df = raw_sample_df.set_index("instance_id", drop=False)

    # Load patches from file (supports JSON/JSONL, dict/list formats)
    patches_to_run = load_patches_from_file(args.predictions_path)
    
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
    
    if missing_instances:
        print(f"Warning: Found {len(missing_instances)} patch instances not in raw sample data:")
        for missing_id in missing_instances[:5]:  # Show first 5
            print(f"  - {missing_id}")
        if len(missing_instances) > 5:
            print(f"  ... and {len(missing_instances) - 5} more")

    if status_counts["empty"] or status_counts["error"]:
        print(
            f"Skipping {status_counts['empty']} empty patches and {status_counts['error']} errored patches before execution"
        )

    print(f"Proceeding with {len(valid_patches)} runnable patches out of {len(patches_to_run)} total patches")

    # Select runtime
    # Auto-detect default platform if not provided: prefer linux/amd64 on Apple Silicon
    detected_platform = None
    if args.use_local_docker and args.docker_platform is None:
        try:
            if py_platform.machine().lower() in {"arm64", "aarch64"}:
                detected_platform = "linux/amd64"
        except Exception:
            detected_platform = None

    eval_fn = eval_with_docker if args.use_local_docker else eval_with_modal

    # Use ThreadPoolExecutor to run evaluations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Create a dictionary mapping futures to their patch samples for progress tracking
        future_to_patch = {
            executor.submit(
                eval_fn,
                patch_text,
                raw_sample_df.loc[patch_sample["instance_id"]],
                str(output_root),
                args.run_id,
                get_prediction_model_name(patch_sample),
                args.dockerhub_username,
                args.scripts_dir,
                prefix=patch_sample.get("prefix", ""),
                redo=args.redo,
                block_network=args.block_network,
                docker_platform=(args.docker_platform or detected_platform) if args.use_local_docker else None,
            ): patch_sample
            for patch_sample, patch_text in valid_patches
        }

        # Track progress with tqdm and show running accuracy
        pbar = tqdm(concurrent.futures.as_completed(future_to_patch), total=len(valid_patches))
        for future in pbar:
            patch_sample = future_to_patch[future]
            instance_id = patch_sample["instance_id"]
            try:
                # Get the result (if any error occurred, it will be raised here)
                output = future.result()
                if output is None:
                    print(f'Evaluation for {instance_id} returned None')
                    patch_status = "error"
                else:
                    raw_sample = raw_sample_df.loc[instance_id]
                    passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
                    f2p = set(eval(raw_sample["fail_to_pass"]))
                    p2p = set(eval(raw_sample["pass_to_pass"]))
                    result = (f2p | p2p) <= passed_tests
                    patch_status = "pass" if result else "fail"
                    eval_results[instance_id] = result

                if patch_status == "error":
                    eval_results[instance_id] = False
                patch_statuses[instance_id] = patch_status
                status_counts[patch_status] += 1

                current_accuracy = status_counts["pass"] / max(1, sum(status_counts.values()))
                pbar.set_description(f"Accuracy: {current_accuracy:.2%}")
            except Exception as exc:
                print(f'Evaluation for {instance_id} generated an exception: {exc}')
                eval_results[instance_id] = False
                patch_statuses[instance_id] = "error"
                status_counts["error"] += 1
                # Update progress bar description with current accuracy
                current_accuracy = status_counts["pass"] / max(1, sum(status_counts.values()))
                pbar.set_description(f"Accuracy: {current_accuracy:.2%}")
    report = build_run_style_report(raw_sample_df, patches_to_run, patch_statuses, eval_results)
    report_path = report_dir / f"{report_model_name}.{args.run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    overall_accuracy = (status_counts["pass"] / len(eval_results)) if eval_results else 0.0
    print("Overall accuracy: ", overall_accuracy)
    print(
        "Summary: \n"
        f"pass={status_counts['pass']}, \n"
        f"fail={status_counts['fail']}, \n"
        f"error={status_counts['error']}, \n"
        f"empty={status_counts['empty']}"
    )
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
