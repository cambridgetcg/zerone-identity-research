#!/usr/bin/env python3
"""Check recorded temporal-memory artifacts without neural inference or downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("temporal_memory_analysis", ROOT / "experiments/temporal_memory/analyze.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
require, sha256 = ANALYSIS.require, ANALYSIS.sha256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_path(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact path")
    resolved = (ROOT / path).resolve()
    require(ROOT in resolved.parents, "Artifact leaves repository")
    return resolved


def provenance(raw, hashes):
    source = raw["provenance"]
    require((source["model"], source["model_revision"]) in ANALYSIS.MODELS, "Unplanned model or revision")
    manifest_path = ("evidence/c2c-fidelity-sources.json" if source["model"] == ANALYSIS.MODELS[0][0]
                     else "evidence/temporal-memory-1.7b-sources.json")
    expected = {
        "harness_sha256": "experiments/temporal_memory/run_experiment.py",
        "branch_helper_sha256": "experiments/checkpoint_branches/run_experiment.py",
        "checkpoint_helper_sha256": "experiments/c2c_fidelity/checkpoint_control.py",
        "fixtures_sha256": "experiments/temporal_memory/fixtures.json",
        "asset_manifest_sha256": manifest_path,
    }
    for field, relative in expected.items():
        require(source[field] == hashes[relative], "Recorded inference source digest differs: " + field)
    manifest = json.loads(repository_path(manifest_path).read_text())
    receivers = [asset for asset in manifest["assets"] if asset["alias"] == "receiver"]
    require(len(receivers) == 1, "Missing receiver model manifest")
    receiver = receivers[0]
    require(source["model"] == receiver["repo"] and source["model_revision"] == receiver["revision"]
            and source["model_files"] == receiver["files"], "Recorded model files differ from pinned manifest")
    seen = set()
    for item in receiver["files"]:
        path = Path(item["path"])
        require(not path.is_absolute() and ".." not in path.parts and item["path"] not in seen,
                "Unsafe or duplicate model asset path")
        seen.add(item["path"])
        ANALYSIS.integer(item["bytes"], 1)
        sha256(item["sha256"])
    runtime = source["runtime"]
    expected_runtime = {"python": "3.13.11", "torch": "2.6.0", "transformers": "4.52.4",
                        "device": "mps", "dtype": "float32", "attention": "eager", "sampling": "greedy",
                        "threads": 6, "seed": 20260921, "automatic_mps_fallback": False,
                        "offline": True, "remote_code": False, "implicit_credentials": False}
    require(all(runtime[key] == value for key, value in expected_runtime.items()), "Recorded apparatus differs from protocol")
    packages = source["package_source_files"]
    require(len(packages) == 5 and {item["path"] for item in packages} == {
        "transformers/cache_utils.py", "transformers/models/qwen3/modeling_qwen3.py",
        "transformers/tokenization_utils_fast.py", "torch/nn/modules/module.py", "torch/version.py"},
        "Unexpected package source scope")
    for item in packages:
        sha256(item["sha256"])


def preflight(raw, cases, formats):
    pre = raw["token_preflight"]
    require(all(pre[field] is True for field in ("passed", "before_model_weight_loading",
                "twin_parent_positions_equal", "gold_annotations_never_rendered")), "Token preflight failed")
    require(pre["thinking"] is False and pre["case_count"] == 24 and pre["formats"] == list(formats),
            "Unexpected preflight formats or case scope")
    ANALYSIS.integer(pre["boundary_checks"], 1)
    ids = pre["digit_token_ids"]
    require(set(ids) == set(ANALYSIS.DIGITS) and len(set(ids.values())) == 10, "Invalid diagnostic digit tokens")
    for token in ids.values():
        ANALYSIS.integer(token)
    counts = pre["parent_token_counts"]
    require(set(counts) == set(cases), "Missing parent token counts")
    for case_id, values in counts.items():
        require(set(values) == {"parent", "structured_summary", "limited_summary"}, "Unexpected parent contexts")
        for count in values.values():
            ANALYSIS.integer(count, 1)
        require(values["parent"] == counts[cases[case_id]["twin_id"]]["parent"], "Twin positions do not align")


def calibration_caches(raw, cases):
    checks = raw["parent_unchanged_checks"]
    expected = {(case_id, format_name) for case_id, case in cases.items() if case["split"] == "calibration"
                for format_name in ANALYSIS.FORMATS}
    require(len(checks) == 24 and {(item["case_id"], item["format"]) for item in checks} == expected,
            "Missing calibration cache nonmutation checks")
    for item in checks:
        require(item["passed"] is True, "Calibration reused a mutated cache")
        sha256(item["parent_sha256"])
        sha256(item["updated_sha256"])


def checkpoints(raw, cases):
    receipts = raw["checkpoints"]
    expected = {case_id for case_id, case in cases.items() if case["split"] == "heldout"}
    require(len(receipts) == 16 and {item["case_id"] for item in receipts} == expected,
            "Incomplete serialized checkpoint receipts")
    for receipt in receipts:
        for field in ("serialized", "reference_serialized"):
            item = receipt[field]
            require(Path(item["filename"]).name == item["filename"], "Checkpoint receipt contains a local path")
            ANALYSIS.integer(item["bytes"], 1)
            sha256(item["sha256"])
        tensors = receipt["raw_tensors"]
        records = tensors["tensors"]
        require(tensors["tensor_count"] == len(records) == 56, "Expected 28 key/value layers")
        require({item["name"] for item in records} == {f"{layer}.{kind}" for layer in range(28) for kind in ("key", "value")},
                "Checkpoint key/value names differ")
        count = raw["token_preflight"]["parent_token_counts"][receipt["case_id"]]["parent"]
        for item in records:
            shape = item["shape"]
            require(len(shape) == 4 and shape[0] == 1 and shape[2] == count and item["dtype"] == "torch.float32",
                    "Checkpoint tensor dtype or prefix position differs")
            for dimension in shape:
                ANALYSIS.integer(dimension, 1)
            sha256(item["sha256"])
        payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        require(tensors["sha256"] == hashlib.sha256(payload).hexdigest(), "Aggregate cache digest differs from tensor receipts")
    by_case = {item["case_id"]: item["raw_tensors"]["sha256"] for item in receipts}
    for item in raw["parent_unchanged_checks"]:
        own = by_case[item["case_id"]]
        if item["stage"] == "prepare":
            require(item["parent_sha256"] == own, "Prepared parent differs from its checkpoint receipt")
            sha256(item["updated_sha256"])
        else:
            twin = by_case[cases[item["case_id"]]["twin_id"]]
            require(item["own_parent_sha256"] == own and item["swapped_parent_sha256"] == twin,
                    "Restored own/twin parent differs from its checkpoint receipt")


def check():
    receipt = json.loads((ROOT / "evidence/temporal-memory-receipt.json").read_text())
    require(receipt["schema"] == "temporal-memory-receipt-v1", "Unexpected receipt schema")
    require(re.fullmatch(r"[0-9a-f]{40}", receipt["protocol_commit"]), "Missing pre-score protocol commit")
    for field in ("training_performed", "private_state_exported"):
        require(receipt[field] is False, "Receipt exceeds authorized experiment scope")
    hashes = receipt["artifact_sha256"]
    required = {
        "experiments/temporal_memory/PROTOCOL.md", "experiments/temporal_memory/fixtures.json",
        "experiments/temporal_memory/run_experiment.py", "experiments/temporal_memory/analyze.py",
        "tools/check_temporal_memory.py", "experiments/checkpoint_branches/run_experiment.py",
        "experiments/c2c_fidelity/checkpoint_control.py", "experiments/c2c_fidelity/requirements-lock.txt",
        "evidence/c2c-fidelity-sources.json",
    }
    for relative, observed in hashes.items():
        require(digest(repository_path(relative)) == sha256(observed), "Artifact digest differs: " + relative)
    require(required <= set(hashes), "Missing required source hashes")
    fixture_path = ROOT / "experiments/temporal_memory/fixtures.json"
    require(digest(fixture_path) == ANALYSIS.FIXTURE_SHA256, "Fixture differs from fixed protocol")
    cases = ANALYSIS.fixture_cases(json.loads(fixture_path.read_text()))
    calibration_files = receipt["calibration_files"]
    require(calibration_files in (["evidence/temporal-memory-calibration-0.6b.json"],
                ["evidence/temporal-memory-calibration-0.6b.json", "evidence/temporal-memory-calibration-1.7b.json"]),
            "Unexpected calibration evidence order")
    require(receipt["new_model_downloads"] is (len(calibration_files) == 2), "Incorrect conditional model-download receipt")
    expected_paths = {"selection_file": "evidence/temporal-memory-selection.json",
                      "results_file": "evidence/temporal-memory-results.json",
                      "summary_file": "evidence/temporal-memory-summary.json"}
    require(all(receipt[key] == value for key, value in expected_paths.items()), "Unexpected evidence locations")
    require(set(calibration_files) | set(expected_paths.values()) <= set(hashes), "Evidence file missing receipt digest")
    command = [sys.executable, "experiments/temporal_memory/analyze.py", "--fixtures", str(fixture_path)]
    for relative in calibration_files:
        raw = json.loads(repository_path(relative).read_text())
        provenance(raw, hashes)
        preflight(raw, cases, ANALYSIS.FORMATS)
        calibration_caches(raw, cases)
        command += ["--calibration", relative]
    raw = json.loads(repository_path(receipt["results_file"]).read_text())
    provenance(raw, hashes)
    preflight(raw, cases, (raw["selection"]["format"],))
    checkpoints(raw, cases)
    command += ["--selection", receipt["selection_file"], "--input", receipt["results_file"],
                "--output", receipt["summary_file"], "--check-existing"]
    subprocess.run(command, cwd=ROOT, check=True, timeout=30)
    print(json.dumps({"temporal_memory_artifacts_checked": True, "neural_inference_rerun": False,
                      "remote_model_bytes_rechecked": False}))


if __name__ == "__main__":
    check()
