#!/usr/bin/env python3
"""Validate recorded E1 evidence using only the standard library.

These checks verify provenance, complete accounting and reproducible coding.
They must not turn a failed fidelity or advisory instrument check into missing
evidence. External tensors and neural inference are not reconstructed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/identity_continuity"
MANIFEST = "evidence/temporal-memory-1.7b-sources.json"
RECEIPT = ROOT / "evidence/identity-continuity-receipt.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid nonnegative integer")
    return value


def sha256(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "Invalid SHA-256 digest")
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_path(relative):
    require(isinstance(relative, str), "Artifact path must be text")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,
            "Unsafe artifact path")
    resolved = (ROOT / path).resolve()
    require(ROOT in resolved.parents, "Artifact leaves repository")
    return resolved


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module("e1_validation_fixtures", STUDY / "fixtures.py")


def fixture_material():
    expected = json.loads(json.dumps(FIXTURES.fixture_record()))
    observed = json.loads((STUDY / "fixtures.json").read_text())
    require(observed == expected, "Fixture JSON differs from literal specification")
    cases = {case["id"]: case for case in FIXTURES.CASES}
    require(len(cases) == 10 and
            [sum(case["split"] == split for case in cases.values())
             for split in ("development", "heldout")] == [4, 6],
            "Unexpected E1 history split")
    prompt_count = 0
    for case in cases.values():
        twin = cases[case["twin_id"]]
        require(twin["twin_id"] == case["id"] and
                twin["pair_id"] == case["pair_id"] and
                twin["split"] == case["split"] and
                twin["object"] == case["paired_object"],
                "Invalid paired history fixture")
        require(FIXTURES.autobiography_text(case, "matched") ==
                FIXTURES.autobiography_text(twin, "swapped"),
                "Account-fixed comparison changes supplied account")
        conditions = FIXTURES.all_conditions(case)
        require(len(conditions) == len({item["condition_id"] for item in conditions}) == 18,
                "Duplicate or missing conditions")
        hidden = {}
        for condition in conditions:
            probes = ("label",) if condition["kind"] == "label" else FIXTURES.PROBE_NAMES
            for probe in probes:
                prompt_count += 1
                text = FIXTURES.continuation_prompt(case, condition, probe)
                if condition["kind"] == "core" and condition["provenance"] == "hidden":
                    key = (condition["autobiography"], probe)
                    hidden.setdefault(key, []).append(text)
        require(all(len(texts) == 2 and texts[0] == texts[1]
                    for texts in hidden.values()),
                "Hidden restore/replay prompts are not identical")
    require(prompt_count == 480, "Unexpected E1 readout count")
    return cases


def validate_provenance(raw, hashes):
    source = raw["provenance"]
    require((source["model"], source["model_revision"]) ==
            ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
            "Model differs from fixed apparatus")
    manifest = json.loads(repository_path(MANIFEST).read_text())
    receivers = [item for item in manifest["assets"] if item["alias"] == "receiver"]
    require(len(receivers) == 1, "Expected one pinned receiver model")
    receiver = receivers[0]
    require(source["model"] == receiver["repo"] and
            source["model_revision"] == receiver["revision"] and
            source["model_files"] == receiver["files"],
            "Model file receipts differ from existing manifest")
    seen = set()
    for item in receiver["files"]:
        path = Path(item["path"])
        require(not path.is_absolute() and ".." not in path.parts and
                item["path"] not in seen, "Unsafe or duplicate model file")
        seen.add(item["path"])
        integer(item["bytes"], 1)
        sha256(item["sha256"])
    expected_runtime = {
        "python": "3.13.11", "torch": "2.6.0", "transformers": "4.52.4",
        "device": "mps", "dtype": "float32", "attention": "eager",
        "sampling": "greedy", "threads": 6, "seed": 20260921,
        "automatic_mps_fallback": False, "offline": True,
        "remote_code": False, "implicit_credentials": False,
    }
    require(all(source["runtime"][key] == value
                for key, value in expected_runtime.items()),
            "Runtime differs from fixed apparatus")
    expected_sources = {
        "harness_sha256": "experiments/identity_continuity/run_experiment.py",
        "temporal_helper_sha256": "experiments/temporal_memory/run_experiment.py",
        "branch_helper_sha256": "experiments/checkpoint_branches/run_experiment.py",
        "checkpoint_helper_sha256": "experiments/c2c_fidelity/checkpoint_control.py",
        "protocol_sha256": "experiments/identity_continuity/PROTOCOL.md",
        "fixtures_sha256": "experiments/identity_continuity/fixtures.py",
        "fixtures_json_sha256": "experiments/identity_continuity/fixtures.json",
        "analysis_sha256": "experiments/identity_continuity/analyze.py",
        "requirements_sha256": "experiments/c2c_fidelity/requirements-lock.txt",
        "asset_manifest_sha256": MANIFEST,
    }
    for field, relative in expected_sources.items():
        require(source[field] == hashes[relative], "Inference source digest differs: " + field)
    packages = source["package_source_files"]
    require(len(packages) == 5 and {item["path"] for item in packages} == {
        "transformers/cache_utils.py", "transformers/models/qwen3/modeling_qwen3.py",
        "transformers/tokenization_utils_fast.py", "torch/nn/modules/module.py",
        "torch/version.py"}, "Unexpected package-source scope")
    for item in packages:
        sha256(item["sha256"])


def token_hash(tokens):
    require(isinstance(tokens, list), "Token stream must be a list")
    for token in tokens:
        integer(token)
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def finite(value, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum), "Invalid numerical receipt")


def generation(row, probe, prefix_length):
    tokens = row["token_ids"]
    token_hash(tokens)
    count = integer(row["actual_steps"], 1)
    limit = FIXTURES.GENERATION_LIMITS[probe]
    require(count == len(tokens) == len(row["steps"]) <= limit and
            row["maximum_steps"] == limit, "Generation budget or step accounting differs")
    token_hash(row["eos_token_ids"])
    stopped = tokens[-1] in row["eos_token_ids"]
    require(not any(token in row["eos_token_ids"] for token in tokens[:-1]),
            "Generation continued past EOS")
    require(row["stopped_on_eos"] is stopped and row["truncated"] is (not stopped),
            "Generation completion status differs")
    require(stopped or count == limit, "Unexplained early generation stop")
    require(row["response_token_ids"] == (tokens[:-1] if stopped else tokens),
            "Response tokens differ from complete generated stream")
    require(isinstance(row["response_text"], str) and isinstance(row["raw_text"], str),
            "Raw generated text missing")
    require(row["final_cache_token_count"] == prefix_length + count,
            "Final generated cache token count differs")
    for index, step in enumerate(row["steps"]):
        require(step["step"] == index and step["token_id"] == tokens[index],
                "Logit step and generated token disagree")
        sha256(step["logits_sha256"])
        finite(step["greedy_probability"], maximum=1)
    sha256(row["trajectory_sha256"])
    require(row["first_logits_sha256"] == row["steps"][0]["logits_sha256"],
            "First-logit digest differs from step receipt")
    finite(row["elapsed_seconds"])


def stage_header(raw, phase, split, hashes):
    require(raw["schema"] == "identity-continuity-stage-v1" and raw["status"] == "complete"
            and raw["phase"] == phase and raw["split"] == split,
            "Unexpected or incomplete evidence stage")
    integer(raw["pid"], 1)
    finite(raw["elapsed_seconds"])
    require(re.fullmatch(r"[0-9a-f]{40}", raw["stage_git_commit"]),
            "Missing source commit for inference stage")
    orchestration = raw["orchestration"]
    require(orchestration["worker_pid"] == raw["pid"] and
            orchestration["parent_pid"] != raw["pid"] and
            orchestration["fresh_worker"] is True and orchestration["returncode"] == 0,
            "Stage did not complete in recorded fresh worker")
    pre = raw["token_preflight"]
    require(pre["passed"] is True and pre["weights_loaded"] is False and
            pre["cases"] == 10 and pre["continuation_boundary_checks"] == 480 and
            pre["hidden_preparation_prompts_equal"] is True and pre["thinking"] is False,
            "Token-only preflight differs from declared apparatus")
    integer(pre["assistant_end_token_id"])
    validate_provenance(raw, hashes)


def histories(raw, cases):
    expected_order = [case["id"] for case in FIXTURES.cases_for_split("all")]
    require([item["case_id"] for item in raw["histories"]] == expected_order,
            "Missing, duplicated or out-of-order parent history")
    require(not any(raw[field] for field in ("rows", "parent_controls",
            "hidden_preparation_comparisons", "parent_unchanged_checks", "intervention_ledger")),
            "Preparation unexpectedly contains readout outcomes")
    end_token = raw["token_preflight"]["assistant_end_token_id"]
    for item in raw["histories"]:
        case = cases[item["case_id"]]
        require(item["pair_id"] == case["pair_id"] and item["split"] == case["split"] and
                item["messages"] == FIXTURES.history_messages(case), "History prompt differs")
        prefill = item["prefill_token_ids"]
        require(bool(prefill), "Empty history prefill")
        generation(item["generated"], "history", len(prefill))
        tokens = item["generated"]["token_ids"]
        closure = [] if tokens[-1] == end_token else [end_token]
        require(item["apparatus_closure_token_ids"] == closure,
                "Unrecorded or unnecessary assistant terminator")
        require((item["apparatus_closure_reason"] is not None) == bool(closure),
                "Assistant-closure reason missing or extraneous")
        chunks = [prefill] + [[token] for token in tokens] + ([closure] if closure else [])
        flattened = [token for chunk in chunks for token in chunk]
        require(item["input_chunks"] == chunks and item["all_token_ids"] == flattened and
                item["all_token_ids_sha256"] == token_hash(flattened) and
                item["parent_token_count"] == len(flattened),
                "Saved history differs from exact generated-token/chunk ledger")
        binary = item["serialized"]
        require(binary["filename"] == case["id"] + ".history.pt", "Unexpected checkpoint filename")
        integer(binary["bytes"], 1)
        sha256(binary["sha256"])
        receipt = item["raw_tensors"]
        tensors = receipt["tensors"]
        names = {f"{layer}.{kind}" for layer in range(28) for kind in ("key", "value")}
        require(receipt["tensor_count"] == len(tensors) == 56 and
                {tensor["name"] for tensor in tensors} == names,
                "Missing checkpoint key/value tensor receipts")
        require([tensor["name"] for tensor in tensors] == sorted(names),
                "Checkpoint tensor receipt order differs")
        for tensor in tensors:
            shape = tensor["shape"]
            require(len(shape) == 4 and shape[0] == 1 and shape[2] == len(flattened) and
                    tensor["dtype"] == "torch.float32", "Checkpoint tensor shape or dtype differs")
            for dimension in shape:
                integer(dimension, 1)
            sha256(tensor["sha256"])
        computed = hashlib.sha256(json.dumps(tensors, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        require(receipt["sha256"] == computed, "Aggregate checkpoint tensor digest differs")
        sha256(item["parent_next_logits_sha256"])
    return {item["case_id"]: item for item in raw["histories"]}


def comparison(record, byte_equal, token_equal=None):
    require(record["logits_bitwise_equal"] is byte_equal,
            "Recorded logit equality disagrees with compared digests")
    require(type(record["logits_values_equal"]) is bool and
            type(record["greedy_token_equal"]) is bool, "Invalid comparison flags")
    finite(record["maximum_absolute_logit_difference"])
    finite(record["total_variation_distance"], maximum=1.000000000001)
    if byte_equal:
        require(record["logits_values_equal"] is True and record["greedy_token_equal"] is True
                and record["maximum_absolute_logit_difference"] == 0 and
                record["total_variation_distance"] == 0, "Contradictory exact-logit comparison")
    if token_equal is not None:
        require(record["greedy_token_equal"] is token_equal,
                "Comparison greedy equality disagrees with generated tokens")


def evaluation(raw, history_raw, history_map, history_digest, analysis):
    split = raw["split"]
    cases = FIXTURES.cases_for_split(split)
    expected_ids = [case["id"] for case in cases]
    require(raw["provenance"] == history_raw["provenance"] and
            raw["token_preflight"] == history_raw["token_preflight"] and
            raw["preparation_sha256"] == history_digest and
            raw["preparation_pid"] == history_raw["pid"] and
            raw["pid"] != history_raw["pid"] and raw["distinct_preparation_process"] is True,
            "Evaluation preparation provenance differs")
    require(not raw["histories"], "Evaluation unexpectedly regenerates parent histories")
    rows = analysis.normalize(raw)
    expected_order = [(case["id"], condition["condition_id"], probe)
                      for case in cases for condition in FIXTURES.all_conditions(case)
                      for probe in (("label",) if condition["kind"] == "label" else FIXTURES.PROBE_NAMES)]
    require([(row["case_id"], row["condition"]["condition_id"], row["probe"]) for row in rows]
            == expected_order, "Readout order differs from counterbalanced fixture")
    require([item["case_id"] for item in raw["parent_controls"]] == expected_ids and
            [item["case_id"] for item in raw["parent_unchanged_checks"]] == expected_ids,
            "Parent control accounting differs")
    controls = {item["case_id"]: item for item in raw["parent_controls"]}
    for case_id, control in controls.items():
        history = history_map[case_id]
        require(control["saved_parent_sha256"] == history["raw_tensors"]["sha256"],
                "Control refers to another saved checkpoint")
        for key in ("restored_parent_sha256", "replayed_parent_sha256"):
            sha256(control[key])
        require(control["restored_replayed_cache_exact"] is
                (control["restored_parent_sha256"] == control["replayed_parent_sha256"]) and
                control["restored_saved_cache_exact"] is
                (control["restored_parent_sha256"] == control["saved_parent_sha256"]),
                "Parent cache equality flags disagree with digests")
        require(control["saved_parent_next_logits_sha256"] == history["parent_next_logits_sha256"],
                "Control saved-logit digest differs")
        sha256(control["replayed_parent_next_logits_sha256"])
        comparison(control["parent_next_logits_comparison"],
                   control["saved_parent_next_logits_sha256"] == control["replayed_parent_next_logits_sha256"])
    require(len(raw["intervention_ledger"]) == len(rows), "Missing intervention ledger entries")
    for row, intervention in zip(rows, raw["intervention_ledger"]):
        condition = row["condition"]
        history = history_map[row["case_id"]]
        fresh = condition["preparation"] == "fresh"
        inputs = row["input_token_ids"]
        require(inputs == row["suffix_token_ids"] and bool(inputs), "Input token aliases differ")
        prefix = [] if fresh else history["all_token_ids"]
        effective_hash = token_hash(prefix + inputs)
        require(row["effective_input_token_ids_sha256"] == effective_hash and
                row["parent_token_count"] == len(prefix), "Effective input receipt differs")
        expected_cache = None if fresh else controls[row["case_id"]][condition["preparation"] + "_parent_sha256"]
        require(row["parent_cache_sha256"] == expected_cache, "Readout parent cache differs")
        generation(row, row["probe"], len(prefix) + len(inputs))
        require(intervention == {
            "case_id": row["case_id"], "condition_id": condition["condition_id"], "probe": row["probe"],
            "preparation": condition["preparation"],
            "history_record_sha256": None if fresh else history["all_token_ids_sha256"],
            "autobiography": condition["autobiography"], "provenance": condition["provenance"],
            "label": condition["label"], "input_token_ids_sha256": token_hash(inputs),
            "effective_input_token_ids_sha256": effective_hash}, "Intervention ledger differs from readout")
    for check in raw["parent_unchanged_checks"]:
        control = controls[check["case_id"]]
        expected = {method: control[method + "_parent_sha256"] for method in ("restored", "replayed")}
        require(check["before"] == expected and set(check["after"]) == set(expected),
                "Parent nonmutation receipt lacks compared hashes")
        for value in check["after"].values():
            sha256(value)
        require(check["passed"] is (check["before"] == check["after"]),
                "Parent nonmutation flag disagrees with hashes")
    hidden_comparisons(raw, rows)
    return analysis.analyze(raw)


def hidden_comparisons(raw, rows):
    lookup = {(row["case_id"], row["condition"]["autobiography"], row["probe"],
               row["condition"]["preparation"]): row for row in rows
              if row["condition"]["kind"] == "core" and row["condition"]["provenance"] == "hidden"}
    expected = {(case["id"], note, probe) for case in FIXTURES.cases_for_split(raw["split"])
                for note in ("matched", "swapped", "absent") for probe in FIXTURES.PROBE_NAMES}
    seen = set()
    for item in raw["hidden_preparation_comparisons"]:
        key = (item["case_id"], item["autobiography"], item["probe"])
        require(key in expected and key not in seen, "Duplicate or unexpected hidden comparison")
        seen.add(key)
        require({item["reference_preparation"], item["observed_preparation"]} == {"restored", "replayed"},
                "Hidden comparison is not a restore/replay pair")
        a = lookup[(*key, item["reference_preparation"])]
        b = lookup[(*key, item["observed_preparation"])]
        ta, tb = a["token_ids"], b["token_ids"]
        n = min(len(ta), len(tb))
        first = next((i for i in range(n) if ta[i] != tb[i]), None)
        comparable = n if first is None else first + 1
        require(item["effective_inputs_equal"] is
                (a["effective_input_token_ids_sha256"] == b["effective_input_token_ids_sha256"]) and
                item["reference_steps"] == len(ta) and item["observed_steps"] == len(tb) and
                item["first_token_divergence_step"] == first and
                item["uncompared_reference_steps"] == len(ta) - comparable and
                item["uncompared_observed_steps"] == len(tb) - comparable and
                item["token_trajectory_equal"] is (ta == tb), "Hidden comparison accounting differs")
        require(item["trajectory_bitwise_equal"] is
                (ta == tb and a["trajectory_sha256"] == b["trajectory_sha256"]),
                "Full-trajectory equality disagrees with receipts")
        steps = item["same_generated_prefix_comparisons"]
        require(len(steps) == comparable and item["first_logits"] == steps[0],
                "Hidden comparison includes unequal generated prefixes or misses decisions")
        for index, record in enumerate(steps):
            require(record["step"] == index, "Hidden decision index differs")
            comparison(record, a["steps"][index]["logits_sha256"] == b["steps"][index]["logits_sha256"],
                       ta[index] == tb[index])
    require(seen == expected, "Missing hidden preparation comparisons")


def check():
    cases = fixture_material()
    receipt = json.loads(RECEIPT.read_text())
    require(receipt["schema"] == "identity-continuity-receipt-v1", "Unexpected E1 receipt schema")
    for field in ("protocol_commit", "history_commit", "development_commit"):
        require(re.fullmatch(r"[0-9a-f]{40}", receipt[field]), "Missing pre-stage commit: " + field)
    for field in ("training_performed", "private_state_exported", "new_model_downloads"):
        require(receipt[field] is False, "Receipt exceeds authorized empirical scope")
    files = {
        "history_file": "evidence/identity-continuity-history.json",
        "development_file": "evidence/identity-continuity-development.json",
        "heldout_file": "evidence/identity-continuity-heldout.json",
        "development_summary_file": "evidence/identity-continuity-development-summary.json",
        "summary_file": "evidence/identity-continuity-summary.json",
    }
    require(all(receipt[field] == relative for field, relative in files.items()),
            "Unexpected E1 evidence paths")
    hashes = receipt["artifact_sha256"]
    required = {
        "experiments/identity_continuity/PROTOCOL.md", "experiments/identity_continuity/fixtures.py",
        "experiments/identity_continuity/fixtures.json", "experiments/identity_continuity/run_experiment.py",
        "experiments/identity_continuity/analyze.py", "tools/check_identity_continuity.py",
        "experiments/temporal_memory/run_experiment.py", "experiments/checkpoint_branches/run_experiment.py",
        "experiments/c2c_fidelity/checkpoint_control.py", "experiments/c2c_fidelity/requirements-lock.txt",
        "evidence/c2c-fidelity-sources.json", MANIFEST, *files.values(),
    }
    require(required <= set(hashes), "Required source or evidence hash missing")
    for relative, expected in hashes.items():
        require(digest(repository_path(relative)) == sha256(expected), "Artifact digest differs: " + relative)
    history_raw = json.loads(repository_path(files["history_file"]).read_text())
    stage_header(history_raw, "prepare", "all", hashes)
    require(history_raw["stage_git_commit"] == receipt["protocol_commit"],
            "History was not generated at recorded protocol commit")
    history_map = histories(history_raw, cases)
    analysis = load_module("e1_validation_analysis", STUDY / "analyze.py")
    for split, source_key, summary_key, commit_key in (
        ("development", "development_file", "development_summary_file", "history_commit"),
        ("heldout", "heldout_file", "summary_file", "development_commit"),
    ):
        raw = json.loads(repository_path(files[source_key]).read_text())
        stage_header(raw, "evaluate", split, hashes)
        require(raw["stage_git_commit"] == receipt[commit_key], "Evaluation source-stage commit differs")
        regenerated = evaluation(raw, history_raw, history_map, hashes[files["history_file"]], analysis)
        stored = json.loads(repository_path(files[summary_key]).read_text())
        require(stored == regenerated, "Stored summary differs from frozen complete-case analysis")
    print(json.dumps({"identity_continuity_artifacts_checked": True,
                      "neural_inference_rerun": False, "external_tensor_bytes_rechecked": False,
                      "positive_fidelity_or_competence_required": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-only", action="store_true",
                        help="Check frozen literal fixtures before inference")
    args = parser.parse_args()
    fixture_material()
    if args.fixtures_only:
        print(json.dumps({"identity_continuity_fixtures_checked": True,
                          "neural_inference_rerun": False}))
        return
    check()


if __name__ == "__main__":
    main()
