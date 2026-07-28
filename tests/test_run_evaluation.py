from swebenchpro.harness.run_evaluation import create_entryscript


def _sample():
    return {
        "before_repo_set_cmd": "echo setup",
        "selected_test_files_to_run": "['tests/test_example.py']",
        "base_commit": "abc123",
    }


def test_create_entryscript_does_not_clean_by_default():
    script = create_entryscript(_sample())

    assert "git clean -fd" not in script


def test_create_entryscript_cleans_before_applying_patch():
    script = create_entryscript(_sample(), clean_start=True)

    assert "git clean -fd" in script
    assert script.index("git checkout abc123") < script.index("git clean -fd")
    assert script.index("git clean -fd") < script.index("git apply --verbose")
