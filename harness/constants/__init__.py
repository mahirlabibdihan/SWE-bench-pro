from pathlib import Path


APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
DOCKER_PATCH = "/workspace/patch.diff"
DOCKER_USER = "root"
DOCKER_WORKDIR = "/app"
LOG_INSTANCE = "run_instance.log"
LOG_REPORT = "report.json"
LOG_TEST_OUTPUT = "test_output.txt"
RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")
UTF8 = "utf-8"
