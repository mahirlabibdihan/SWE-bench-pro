"""Run-report generation for SWE-bench Pro."""


def make_run_report(raw_sample_df, patches_to_run, patch_statuses):
    total_ids = set(raw_sample_df.index.tolist())
    submitted_ids = {
        patch.get("instance_id")
        for patch in patches_to_run
        if isinstance(patch, dict) and patch.get("instance_id")
    }
    submitted_ids &= total_ids

    completed_ids = {iid for iid, status in patch_statuses.items() if status in {"pass", "fail"}}
    resolved_ids = {iid for iid, status in patch_statuses.items() if status == "pass"}
    unresolved_ids = {iid for iid, status in patch_statuses.items() if status == "fail"}
    empty_patch_ids = {iid for iid, status in patch_statuses.items() if status == "empty"}
    error_ids = {iid for iid, status in patch_statuses.items() if status == "error"}

    return {
        "total_instances": len(total_ids),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(total_ids - submitted_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "submitted_ids": sorted(submitted_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
        "schema_version": 2,
    }
