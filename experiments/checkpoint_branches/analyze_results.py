#!/usr/bin/env python3
"""Validate and summarize recorded checkpoint branches with the standard library.

This verifies artifact consistency and recomputes descriptive arithmetic. It does
not rerun inference, authenticate model bytes, or establish subjective continuity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


BRANCHES = ("left", "right")
KINDS = ("original", "persistent", "current")
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ARMS = ("restored", "same_chunk_replay", "full_replay", "swapped_parent", "redacted_parent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, location="input"):
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{location}[{index}]")


def number(value, name, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid number: {name}")
    require(minimum is None or value >= minimum, f"Below allowed minimum: {name}")
    require(maximum is None or value <= maximum, f"Above allowed maximum: {name}")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer: {name}")
    return value


def boolean(value, name):
    require(type(value) is bool, f"Invalid Boolean: {name}")
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            f"Invalid SHA-256: {name}")
    return value


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9),
            f"Inconsistent numeric fields: {name}")


def fixture_cases(fixture):
    require(fixture["schema"] == "checkpoint-branch-fixtures-v1", "Unexpected fixture schema")
    require(fixture["case_count"] == 8 and fixture["branch_probe_count"] == 48,
            "Unexpected fixture counts")
    require(isinstance(fixture["instruction"], str) and fixture["instruction"], "Missing instruction")
    require(fixture["redaction"] == "UNKNOWN", "Unexpected fixed redaction")
    cases = {}
    probes = {}
    for case in fixture["cases"]:
        case_id = case["id"]
        require(isinstance(case_id, str) and case_id and case_id not in cases, "Duplicate or invalid case")
        cases[case_id] = case
        for field in ("parent_text", "redacted_parent_text"):
            require(isinstance(case[field], str) and case[field], f"Missing {field}")
        require(len(case["branches"]) == 2
                and {b["id"] for b in case["branches"]} == set(BRANCHES), "Incomplete fixture branches")
        for branch in case["branches"]:
            require(isinstance(branch["update_text"], str) and branch["update_text"], "Missing update")
            require(len(branch["probes"]) == 3
                    and {p["kind"] for p in branch["probes"]} == set(KINDS), "Incomplete fixture probes")
            for probe in branch["probes"]:
                require(probe["id"] == probe["kind"], "Fixed probe ID must equal its kind")
                require(probe["correct"] in ("A", "B"), "Invalid fixture gold")
                for field in ("question", "choice_A", "choice_B"):
                    require(isinstance(probe[field], str) and probe[field], f"Missing probe {field}")
                require(probe["choice_A"] != probe["choice_B"], "Indistinguishable choices")
                probes[case_id, branch["id"], probe["id"]] = probe
        require(case["control_probe"] == probes[case_id, "left", "original"], "Unexpected control probe")
    require(len(cases) == 8 and len(probes) == 48, "Expected eight parents and 48 probes")
    require(Counter(p["correct"] for p in probes.values()) == {"A": 24, "B": 24},
            "Expected balanced branch gold")
    for case_id, case in cases.items():
        twin_id = case["twin_id"]
        require(twin_id in cases and twin_id != case_id and cases[twin_id]["twin_id"] == case_id,
                "Broken counterfactual twin relation")
        require(case["parent_text"] != cases[twin_id]["parent_text"], "Twins have identical parent history")
        require(case["redacted_parent_text"] == cases[twin_id]["redacted_parent_text"],
                "Redacted twins differ")
        updates = {b["id"]: b["update_text"] for b in case["branches"]}
        twin_updates = {b["id"]: b["update_text"] for b in cases[twin_id]["branches"]}
        require(updates == twin_updates and updates["left"] != updates["right"], "Unexpected branch updates")
        for branch in BRANCHES:
            for kind in KINDS:
                probe, twin = probes[case_id, branch, kind], probes[twin_id, branch, kind]
                require(all(probe[f] == twin[f] for f in ("id", "kind", "question", "choice_A", "choice_B")),
                        "Twin question or options differ")
                require((probe["correct"] == twin["correct"]) == (kind == "current"),
                        "Twin gold does not match historical/current design")
        for kind in ("original", "persistent"):
            require(probes[case_id, "left", kind] == probes[case_id, "right", kind],
                    "Historical probe differs across later branches")
        require(probes[case_id, "left", "current"]["correct"]
                != probes[case_id, "right", "current"]["correct"], "Current answers do not differ by branch")
    return cases, probes


def mean(values):
    return math.fsum(values) / len(values) if values else None


def aggregate(observations, supplied=False):
    gold_field = "supplied_history_correct" if supplied else "correct"
    forced = sum(row["forced_choice"] == row[gold_field] for row in observations)
    greedy = sum(row["greedy_choice"] == row[gold_field] for row in observations)
    return {
        "observations": len(observations),
        "forced_correct": forced,
        "forced_accuracy": forced / len(observations) if observations else None,
        "greedy_correct": greedy,
        "greedy_accuracy": greedy / len(observations) if observations else None,
        "mean_unconditional_ab_mass": mean([r["unconditional_ab_mass"] for r in observations]),
        "mean_forced_probability_correct": mean([r["forced_ab_probability"][r[gold_field]] for r in observations]),
        "mean_unconditional_probability_correct": mean([r["unconditional_ab_probability"][r[gold_field]] for r in observations]),
        "mean_unconditional_logprob_correct": mean([r["absolute_log_probabilities"][r[gold_field]] for r in observations]),
        "forced_choice_counts": dict(Counter(r["forced_choice"] for r in observations)),
        "greedy_label_counts": dict(Counter(r["greedy_choice"] or "other" for r in observations)),
    }


def validate_score(row, correct, option_ids, supplied_correct=None):
    require(row["correct"] == correct, "Recorded gold differs from the fixture")
    forced, absolute = row["forced_ab_probability"], row["unconditional_ab_probability"]
    require(set(forced) == set(absolute) == {"A", "B"}, "Expected exactly A/B probabilities")
    for label in ("A", "B"):
        number(forced[label], "forced probability", 0, 1)
        number(absolute[label], "absolute probability", 0, 1)
    close(sum(forced.values()), 1, "forced probabilities sum")
    mass = number(row["unconditional_ab_mass"], "A/B mass", 0, 1.0000002)
    close(mass, sum(absolute.values()), "A/B mass sum")
    if mass > 1e-30:
        for label in ("A", "B"):
            close(forced[label], absolute[label] / mass, "conditional vs absolute probability")
    choice = "A" if forced["A"] >= forced["B"] else "B"
    require(row["forced_choice"] == choice, "Forced choice disagrees with probabilities")
    require(boolean(row["forced_correct"], "forced correctness") == (choice == correct),
            "Inconsistent forced correctness")
    close(number(row["forced_probability_correct"], "gold forced probability", 0, 1),
          forced[correct], "gold forced probability")
    probability = number(row["unconditional_probability_correct"], "gold absolute probability", 0, 1)
    close(probability, absolute[correct], "gold absolute probability")
    log_probability = number(row["unconditional_logprob_correct"], "gold log probability", maximum=1e-7)
    close(math.exp(log_probability), probability, "gold probability and log probability")
    greedy_id = integer(row["greedy_next_token_id"], "greedy token ID")
    greedy_choice = next((label for label, token in option_ids.items() if token == greedy_id), None)
    require(row["greedy_choice"] == greedy_choice, "Greedy choice disagrees with token ID")
    require(boolean(row["greedy_correct"], "greedy correctness") == (greedy_choice == correct),
            "Inconsistent greedy correctness")
    require(greedy_choice is None or greedy_choice == choice,
            "Unrestricted greedy option differs from highest-probability A/B option")
    require(isinstance(row["greedy_next_token_text"], str), "Missing decoded greedy token")
    if greedy_choice is not None:
        require(row["greedy_next_token_text"] == greedy_choice, "Decoded A/B token disagrees with its ID")
    if supplied_correct is not None:
        require(row["supplied_history_correct"] == supplied_correct, "Incorrect supplied-history gold")
        if "supplied_history_forced_correct" in row:
            require(boolean(row["supplied_history_forced_correct"], "supplied forced correctness")
                    == (choice == supplied_correct), "Incorrect supplied-history forced correctness")
        if "supplied_history_greedy_correct" in row:
            require(boolean(row["supplied_history_greedy_correct"], "supplied greedy correctness")
                    == (greedy_choice == supplied_correct), "Incorrect supplied-history greedy correctness")
    if "latency_seconds" in row:
        number(row["latency_seconds"], "latency", 0)


def validate_comparison(comparison):
    maximum = number(comparison["maximum_absolute_logit_difference"], "maximum logit difference", 0)
    variation = number(comparison["total_variation_distance"], "total variation", 0, 1.0000002)
    greedy_equal = boolean(comparison["greedy_token_equal"], "greedy token agreement")
    exact = boolean(comparison["logits_exactly_equal"], "exact logit agreement")
    require(not exact or maximum == 0, "Byte-equal logits have a nonzero maximum difference")
    require(not exact or (variation == 0 and greedy_equal), "Exact logits have unequal distributions or tokens")
    if "logits_values_equal" in comparison:
        require(boolean(comparison["logits_values_equal"], "numeric logit agreement") == (maximum == 0),
                "Numeric equality flag disagrees with maximum difference")


def comparison_summary(comparisons):
    return {
        "comparisons": len(comparisons),
        "exact_logit_comparisons": sum(c["logits_exactly_equal"] for c in comparisons),
        "greedy_token_agreements": sum(c["greedy_token_equal"] for c in comparisons),
        "maximum_absolute_logit_difference": max(
            (c["maximum_absolute_logit_difference"] for c in comparisons), default=None),
        "maximum_total_variation_distance": max(
            (c["total_variation_distance"] for c in comparisons), default=None),
        "mean_total_variation_distance": mean([c["total_variation_distance"] for c in comparisons]),
    }


def branch_success(observations, cases, supplied=False):
    by_key = {(r["case_id"], r["branch_id"], r["kind"]): r for r in observations}
    gold = "supplied_history_correct" if supplied else "correct"
    result = {"parents": len(cases), "branches": 2 * len(cases)}
    for mode in ("forced", "greedy"):
        choice = mode + "_choice"
        result[f"both_branches_current_{mode}_correct_parents"] = sum(all(
            by_key[case_id, branch, "current"][choice] == by_key[case_id, branch, "current"][gold]
            for branch in BRANCHES) for case_id in cases)
        result[f"all_three_{mode}_correct_branches"] = sum(all(
            by_key[case_id, branch, kind][choice] == by_key[case_id, branch, kind][gold]
            for kind in KINDS) for case_id in cases for branch in BRANCHES)
        result[f"all_six_{mode}_correct_parents"] = sum(all(
            by_key[case_id, branch, kind][choice] == by_key[case_id, branch, kind][gold]
            for branch in BRANCHES for kind in KINDS) for case_id in cases)
    return result


def summarize_scores(cases, parent_references, branch_references, rows):
    by_arm = {arm: [row for row in rows.values() if row["arm"] == arm] for arm in ARMS}
    parents = list(parent_references.values())
    references = list(branch_references.values())
    competent = sorted((row["case_id"], row["kind"]) for row in parents if row["forced_correct"])
    competent_set = set(competent)
    result = {
        "parent_references": {
            "all": aggregate(parents),
            "by_kind": {kind: aggregate([r for r in parents if r["kind"] == kind])
                        for kind in ("original", "persistent")},
            "both_historical_probes_forced_correct_parents": sum(all(
                parent_references[case_id, kind]["forced_correct"]
                for kind in ("original", "persistent")) for case_id in cases),
            "both_historical_probes_greedy_correct_parents": sum(all(
                parent_references[case_id, kind]["greedy_correct"]
                for kind in ("original", "persistent")) for case_id in cases),
        },
        "uninterrupted_branch_references": {
            "all": aggregate(references),
            "by_kind": {kind: aggregate([r for r in references if r["kind"] == kind]) for kind in KINDS},
            "branch_success": branch_success(references, cases),
        },
        "arms": {arm: {
            "all": aggregate(observations),
            "by_kind": {kind: aggregate([r for r in observations if r["kind"] == kind]) for kind in KINDS},
            "branch_success": branch_success(observations, cases),
        } for arm, observations in by_arm.items()},
        "swapped_parent_against_supplied_history": {
            "all": aggregate(by_arm["swapped_parent"], supplied=True),
            "by_kind": {kind: aggregate([r for r in by_arm["swapped_parent"] if r["kind"] == kind], supplied=True)
                        for kind in KINDS},
            "branch_success": branch_success(by_arm["swapped_parent"], cases, supplied=True),
        },
        "parent_competent_historical_subset": {
            "criterion": "Original parent-reference forced A/B choice correct for this case and historical question; not a requirement for primary inclusion.",
            "parent_question_count": len(competent),
            "branch_question_count_per_arm": 2 * len(competent),
            "parent_questions": [{"case_id": case_id, "kind": kind} for case_id, kind in competent],
            "arms": {arm: aggregate([r for r in observations if (r["case_id"], r["kind"]) in competent_set])
                     for arm, observations in by_arm.items()},
            "by_kind": {kind: {
                "parent_question_count": sum(k == kind for _, k in competent),
                "arms": {arm: aggregate([r for r in observations if r["kind"] == kind
                                         and (r["case_id"], kind) in competent_set])
                         for arm, observations in by_arm.items()},
            } for kind in ("original", "persistent")},
            "restored_prior_correct_now_forced_incorrect": [
                {"case_id": r["case_id"], "branch": r["branch_id"], "kind": r["kind"],
                 "correct": r["correct"], "forced_choice": r["forced_choice"]}
                for r in by_arm["restored"] if (r["case_id"], r["kind"]) in competent_set and not r["forced_correct"]],
        },
    }
    contrasts = {}
    for kind in KINDS:
        restored = [r for r in by_arm["restored"] if r["kind"] == kind]
        swapped = [rows["swapped_parent", r["case_id"], r["branch_id"], kind] for r in restored]
        contrasts[kind] = {
            "observations": len(restored),
            "changed_forced_choices": sum(a["forced_choice"] != b["forced_choice"] for a, b in zip(restored, swapped)),
            "changed_greedy_token_ids": sum(a["greedy_next_token_id"] != b["greedy_next_token_id"] for a, b in zip(restored, swapped)),
            "mean_intended_gold_logprob_restored_minus_swapped": mean([
                a["unconditional_logprob_correct"] - b["unconditional_logprob_correct"] for a, b in zip(restored, swapped)]),
            "mean_intended_gold_forced_probability_restored_minus_swapped": mean([
                a["forced_probability_correct"] - b["forced_probability_correct"] for a, b in zip(restored, swapped)]),
        }
    result["restored_versus_swapped_behavior"] = contrasts
    return result


def normalize_score(raw):
    """Use common analytical names without changing the recorded runner schema."""
    row = dict(raw)
    row["kind"] = raw["probe"]
    if "branch" in raw:
        row["branch_id"] = raw["branch"]
    aliases = {
        "forced_ab_probability": "conditional_probabilities",
        "unconditional_ab_probability": "absolute_probabilities",
        "unconditional_ab_mass": "absolute_ab_mass",
        "greedy_next_token_id": "greedy_token_id",
        "greedy_next_token_text": "greedy_token_text",
        "greedy_choice": "greedy_label",
        "supplied_history_forced_correct": "forced_supplied_history_correct",
        "supplied_history_greedy_correct": "greedy_supplied_history_correct",
    }
    for alias, name in aliases.items():
        row[alias] = raw[name]
    correct = raw["correct"]
    require(correct in ("A", "B"), "Invalid raw gold label")
    row["forced_probability_correct"] = raw["conditional_probabilities"][correct]
    row["unconditional_probability_correct"] = raw["absolute_probabilities"][correct]
    row["unconditional_logprob_correct"] = raw["absolute_log_probabilities"][correct]
    require(set(raw["absolute_log_probabilities"]) == {"A", "B"}, "Missing absolute label log probabilities")
    for label in ("A", "B"):
        log_probability = number(raw["absolute_log_probabilities"][label], "absolute label log probability", maximum=1e-7)
        close(math.exp(log_probability), raw["absolute_probabilities"][label], "absolute label log probability")
    digest(raw["logits_sha256"], "scored logits")
    if "query_elapsed_seconds" in raw:
        number(raw["query_elapsed_seconds"], "query elapsed seconds", 0)
    return row


def normalize_comparison(raw):
    value = dict(raw)
    value["logits_exactly_equal"] = raw["logits_bitwise_equal"]
    validate_comparison(value)
    return value


def validate_provenance(report, fixture_bytes, cases):
    provenance = report["provenance"]
    require(provenance["model"] == "Qwen/Qwen3-0.6B"
            and provenance["model_revision"] == MODEL_REVISION, "Unexpected model provenance")
    require(provenance["fixtures_sha256"] == hashlib.sha256(fixture_bytes).hexdigest(), "Fixture digest mismatch")
    for key in ("harness_sha256", "helper_sha256", "asset_manifest_sha256"):
        digest(provenance[key], key)
    require(len(provenance["model_files"]) > 0, "Missing model file receipts")
    paths = set()
    for item in provenance["model_files"]:
        require(isinstance(item["path"], str) and item["path"] not in paths
                and not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts,
                "Invalid or repeated model file path")
        paths.add(item["path"])
        integer(item["bytes"], "model file size", 1)
        digest(item["sha256"], "model file")
    require("model.safetensors" in paths, "Missing model weight receipt")
    source_paths = set()
    for item in provenance["package_source_files"]:
        require(isinstance(item["path"], str) and item["path"] not in source_paths
                and not Path(item["path"]).is_absolute(), "Invalid package source path")
        source_paths.add(item["path"])
        digest(item["sha256"], "package source")
    require(source_paths == {
        "transformers/cache_utils.py", "transformers/models/qwen3/modeling_qwen3.py",
        "transformers/tokenization_utils_fast.py", "torch/nn/modules/module.py", "torch/version.py",
    }, "Unexpected selected inference source receipt scope")
    runtime = provenance["runtime"]
    require(runtime["torch"].split("+")[0] == "2.6.0" and runtime["transformers"] == "4.52.4",
            "Unexpected inference package versions")
    require(runtime["device"] in ("cpu", "mps") and runtime["dtype"] == "float32"
            and runtime["attention"] == "eager" and runtime["sampling"] == "greedy", "Unexpected runtime settings")
    require(runtime["threads"] == 6 and runtime["seed"] == 20260921, "Unexpected fixed execution settings")
    for key, expected in (("automatic_mps_fallback", False), ("offline", True),
                          ("remote_code", False), ("implicit_credentials", False)):
        require(runtime[key] is expected, "Unexpected runtime boundary: " + key)
    preflight = report["token_preflight"]
    for key in ("passed", "before_model_weight_loading", "twin_and_redacted_positions_equal",
                "redacted_twins_identical", "gold_never_rendered"):
        require(preflight[key] is True, "Token preflight did not pass: " + key)
    require(preflight["thinking"] is False and preflight["case_count"] == 8
            and preflight["branch_question_count"] == 48, "Unexpected token preflight design")
    integer(preflight["boundary_checks"], "token boundary checks", 1)
    require(preflight["labels"] == {"A": 32, "B": 33}, "Unexpected fixed single-token options")
    require(preflight["control_suffix_text"] == "Describe the original record in a short paragraph of about 50 words."
            and preflight["control_maximum_new_tokens"] == 32, "Unexpected continuation control")
    require(isinstance(preflight["chat_opening"], str) and isinstance(preflight["chat_closing"], str),
            "Missing recorded chat boundaries")
    for key in ("parent_token_counts", "parent_token_ids", "redacted_parent_token_ids"):
        require(set(preflight[key]) == set(cases), "Missing or extra preflight parents")
    for case_id, case in cases.items():
        ids = preflight["parent_token_ids"][case_id]
        redacted = preflight["redacted_parent_token_ids"][case_id]
        twin = case["twin_id"]
        require(isinstance(ids, list) and isinstance(redacted, list), "Token IDs must be lists")
        for token in ids + redacted:
            integer(token, "prompt token ID")
        require(len(ids) == len(redacted) == preflight["parent_token_counts"][case_id]
                == len(preflight["parent_token_ids"][twin]) > 0, "Unaligned checkpoint token positions")
        require(ids != preflight["parent_token_ids"][twin], "Twin token IDs do not differ")
        require(redacted == preflight["redacted_parent_token_ids"][twin], "Redacted twin tokens differ")
    stages = report["stages"]
    require(len(stages) == 3 and [s["stage"] for s in stages] == ["prepare", "left", "right"],
            "Expected three ordered experiment stages")
    pids = [integer(report["orchestrator_pid"], "orchestrator PID", 1)]
    for stage in stages:
        pids.append(integer(stage["pid"], "stage PID", 1))
        require(stage["status"] == "complete" and stage["returncode"] == 0, "Experiment subprocess did not complete")
        number(stage["elapsed_seconds"], "stage elapsed seconds", 0)
    require(len(set(pids)) == 4 and report["distinct_stage_processes"] is True,
            "Expected distinct orchestrator and three stage processes")
    number(report["elapsed_seconds"], "experiment elapsed seconds", 0)


def validate_checkpoints(report, cases):
    checkpoints = {}
    for receipt in report["checkpoints"]:
        case_id = receipt["case_id"]
        require(case_id in cases and case_id not in checkpoints, "Unknown or duplicate checkpoint")
        checkpoints[case_id] = receipt
        length = integer(receipt["prefix_length"], "checkpoint prefix length", 1)
        require(length == report["token_preflight"]["parent_token_counts"][case_id], "Checkpoint prefix length mismatch")
        for field, expected_suffix in (("serialized", ".checkpoint.pt"), ("baseline_serialized", ".baseline.pt")):
            artifact = receipt[field]
            require(artifact["filename"] == case_id + expected_suffix, "Unexpected checkpoint artifact filename")
            integer(artifact["bytes"], "serialized artifact bytes", 1)
            digest(artifact["sha256"], "serialized artifact")
        raw = receipt["raw_tensors"]
        require(raw["tensor_count"] == len(raw["tensors"]) == 56, "Expected 56 saved K/V tensors")
        expected_names = {f"{i}.{kind}" for i in range(28) for kind in ("key", "value")}
        require({t["name"] for t in raw["tensors"]} == expected_names, "Missing or extra cache tensor names")
        require(raw["tensors"] == sorted(raw["tensors"], key=lambda item: item["name"]), "Tensor receipt ordering differs")
        for tensor in raw["tensors"]:
            shape = tensor["shape"]
            require(isinstance(shape, list) and len(shape) == 4
                    and all(type(v) is int and v > 0 for v in shape), "Invalid cache tensor shape")
            require(shape[0] == 1 and shape[2] == length and tensor["dtype"] == "torch.float32",
                    "Cache tensor shape/dtype differs from checkpoint boundary")
            digest(tensor["sha256"], "cache tensor")
        expected_hash = hashlib.sha256(json.dumps(raw["tensors"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(raw["sha256"] == expected_hash, "Aggregate raw-tensor receipt digest mismatch")
    require(set(checkpoints) == set(cases), "Incomplete parent checkpoints")
    workers = {}
    for check in report["worker_checkpoint_checks"]:
        key = check["stage"], check["case_id"]
        require(key[0] in BRANCHES and key[1] in cases and key not in workers, "Unknown or duplicate worker checkpoint check")
        workers[key] = check
        for field in ("serialized_bytes_verified", "raw_tensors_verified", "runtime_metadata_verified"):
            require(check[field] is True, "Worker did not verify saved checkpoint: " + field)
        require(check["own_parent_sha256"] == checkpoints[key[1]]["raw_tensors"]["sha256"], "Worker restored a different own parent")
        require(check["swapped_parent_sha256"] == checkpoints[cases[key[1]]["twin_id"]]["raw_tensors"]["sha256"],
                "Worker restored a different swapped parent")
    require(set(workers) == {(branch, case_id) for branch in BRANCHES for case_id in cases}, "Incomplete worker checkpoint checks")
    unchanged = {}
    for check in report["parent_unchanged_checks"]:
        key = check["stage"], check["case_id"]
        require(key[0] in ("prepare", *BRANCHES) and key[1] in cases and key not in unchanged,
                "Unknown or duplicate parent-isolation check")
        unchanged[key] = check
        require(check["passed"] is True, "Reusable parent cache was mutated")
        if key[0] in BRANCHES:
            require(check["own_and_swapped_checked"] is True, "Worker omitted swapped-parent isolation check")
    require(set(unchanged) == {(stage, case_id) for stage in ("prepare", *BRANCHES) for case_id in cases},
            "Incomplete parent-isolation checks")
    branches = {}
    receipt_arms = ("uninterrupted", "restored", "same_chunk_replay", "swapped_parent", "redacted_parent")
    for receipt in report["branch_cache_receipts"]:
        key = receipt["arm"], receipt["case_id"], receipt["branch"]
        require(key[0] in receipt_arms and key[1] in cases and key[2] in BRANCHES and key not in branches,
                "Unknown or duplicate branch cache receipt")
        branches[key] = receipt
        digest(receipt["parent_sha256"], "branch parent")
        digest(receipt["updated_sha256"], "updated branch")
        integer(receipt["updated_length"], "updated branch length", checkpoints[key[1]]["prefix_length"] + 1)
        if key[0] in ("uninterrupted", "restored"):
            require(receipt["parent_sha256"] == checkpoints[key[1]]["raw_tensors"]["sha256"], "Branch has incorrect checkpoint ancestry")
        elif key[0] == "swapped_parent":
            require(receipt["parent_sha256"] == checkpoints[cases[key[1]]["twin_id"]]["raw_tensors"]["sha256"],
                    "Swapped branch has incorrect checkpoint ancestry")
    require(set(branches) == {(arm, case_id, branch) for arm in receipt_arms for case_id in cases for branch in BRANCHES},
            "Incomplete branch cache receipts")
    for case_id, case in cases.items():
        for branch in BRANCHES:
            require(len({branches[arm, case_id, branch]["updated_length"] for arm in receipt_arms}) == 1,
                    "Control branches use unequal positions")
            redacted = branches["redacted_parent", case_id, branch]
            twin_redacted = branches["redacted_parent", case["twin_id"], branch]
            require(redacted == {**twin_redacted, "case_id": case_id}, "Identical redacted branches have different cache receipts")
    return checkpoints, branches


def validate_continuations(report, cases):
    def trajectory(item):
        steps = integer(item["actual_steps"], "actual continuation steps", 1)
        require(steps <= 32 and item["maximum_steps"] == 32, "Continuation exceeds fixed bound")
        require(isinstance(item["token_ids"], list) and len(item["token_ids"]) == steps, "Continuation token count mismatch")
        for token in item["token_ids"]:
            integer(token, "continuation token")
        require(isinstance(item["text"], str), "Missing decoded continuation")
        eos = item["eos_token_ids"]
        require(isinstance(eos, list) and eos == sorted(set(eos)), "Invalid EOS list")
        for token in eos:
            integer(token, "EOS token")
        ended = item["token_ids"][-1] in eos
        require(boolean(item["stopped_on_eos"], "EOS stop") == ended, "EOS stop flag differs from final token")
        require(not any(t in eos for t in item["token_ids"][:-1]), "Generation continued after EOS")
        require(ended or steps == 32, "Continuation stopped without EOS or reaching its bound")
        digest(item["logits_sha256"], "continuation logits")
        return steps
    baselines = {}
    for item in report["continuation_baselines"]:
        case_id = item["case_id"]
        require(case_id in cases and case_id not in baselines, "Unknown or duplicate continuation baseline")
        baselines[case_id] = item
        trajectory(item)
    require(set(baselines) == set(cases), "Incomplete continuation baselines")
    controls = {}
    for raw in report["continuation_controls"]:
        item = dict(raw)
        key = item["case_id"], item["branch"]
        require(key[0] in cases and key[1] in BRANCHES and key not in controls, "Unknown or duplicate continuation control")
        controls[key] = item
        steps = trajectory(item)
        baseline = baselines[key[0]]
        reference_steps = baseline["actual_steps"]
        require(item["reference_steps"] == reference_steps and item["eos_token_ids"] == baseline["eos_token_ids"],
                "Continuation uses a different reference or stop rule")
        comparisons = [normalize_comparison(c) for c in item["step_comparisons"]]
        item["step_comparisons"] = comparisons
        require(len(comparisons) == min(steps, reference_steps), "Incomplete full-logit step comparisons")
        token_equal = item["token_ids"] == baseline["token_ids"]
        require(boolean(item["token_trajectory_equal"], "trajectory agreement") == token_equal, "Trajectory agreement flag differs from tokens")
        for index, comp in enumerate(comparisons):
            require(comp["greedy_token_equal"] == (item["token_ids"][index] == baseline["token_ids"][index]),
                    "Step greedy agreement differs from generated tokens")
        exact = steps == reference_steps and token_equal and all(c["logits_exactly_equal"] for c in comparisons)
        require(boolean(item["exact_restoration"], "exact restoration") == exact, "Exact-restoration flag differs from observations")
        require((item["logits_sha256"] == baseline["logits_sha256"]) == exact,
                "Full continuation hash disagrees with exact-restoration observations")
        require(item["uncompared_reference_steps"] == max(0, reference_steps - steps)
                and item["uncompared_restored_steps"] == max(0, steps - reference_steps), "Incorrect unpaired step counts")
    require(set(controls) == {(case_id, branch) for case_id in cases for branch in BRANCHES}, "Incomplete continuation controls")
    return baselines, controls


def validate_report(report, fixture, fixture_bytes):
    finite_tree(report)
    require(report["schema"] == "checkpoint-branches-results-v1" and report["status"] == "complete",
            "Expected a complete checkpoint-branch run")
    cases, probes = fixture_cases(fixture)
    validate_provenance(report, fixture_bytes, cases)
    checkpoints, cache_receipts = validate_checkpoints(report, cases)
    baselines, controls = validate_continuations(report, cases)
    options = report["token_preflight"]["labels"]
    parents, references, rows = {}, {}, {}
    for raw in report["parent_references"]:
        row = normalize_score(raw)
        key = row["case_id"], row["kind"]
        require(key[0] in cases and key[1] in ("original", "persistent") and key not in parents,
                "Unknown or duplicate parent reference")
        require(row["arm"] == "parent_uninterrupted", "Unexpected parent-reference arm")
        gold = probes[key[0], "left", key[1]]["correct"]
        validate_score(row, gold, options, gold)
        parents[key] = row
    require(set(parents) == {(case_id, kind) for case_id in cases for kind in ("original", "persistent")},
            "Incomplete parent competence references")
    for raw in report["branch_references"]:
        row = normalize_score(raw)
        key = row["case_id"], row["branch_id"], row["kind"]
        require(key in probes and key not in references, "Unknown or duplicate branch reference")
        require(row["arm"] == "uninterrupted", "Unexpected branch-reference arm")
        gold = probes[key]["correct"]
        validate_score(row, gold, options, gold)
        references[key] = row
    require(set(references) == set(probes), "Incomplete uninterrupted branch references")
    for raw in report["rows"]:
        row = normalize_score(raw)
        probe_key = row["case_id"], row["branch_id"], row["kind"]
        key = row["arm"], *probe_key
        require(row["arm"] in ARMS and probe_key in probes and key not in rows,
                "Unknown or duplicate arm observation")
        gold = probes[probe_key]["correct"]
        supplied_gold = gold
        if row["kind"] != "current":
            if row["arm"] == "swapped_parent":
                supplied_gold = probes[cases[row["case_id"]]["twin_id"], row["branch_id"], row["kind"]]["correct"]
            elif row["arm"] == "redacted_parent":
                supplied_gold = None
        validate_score(row, gold, options, supplied_gold)
        require(row["supplied_history_correct"] == supplied_gold, "Incorrect supplied-history label")
        if supplied_gold is None:
            require(row["supplied_history_forced_correct"] is None and row["supplied_history_greedy_correct"] is None,
                    "Redacted old facts must have unknown supplied-history correctness")
        row["baseline_comparison"] = normalize_comparison(raw["baseline_comparison"])
        row["restored_comparison"] = normalize_comparison(raw["restored_comparison"])
        rows[key] = row
    require(set(rows) == {(arm, *key) for arm in ARMS for key in probes}, "Incomplete five-arm/probe cross product")
    independent_fields = ("forced_ab_probability", "unconditional_ab_probability", "unconditional_ab_mass",
                          "forced_choice", "greedy_next_token_id", "greedy_next_token_text", "greedy_choice", "logits_sha256")
    for (arm, case_id, branch, kind), row in rows.items():
        comparisons = ((row["baseline_comparison"], references[case_id, branch, kind]),
                       (row["restored_comparison"], rows["restored", case_id, branch, kind]))
        for comparison, reference in comparisons:
            exact = row["logits_sha256"] == reference["logits_sha256"]
            require(comparison["logits_exactly_equal"] == exact, "Comparison byte-equality flag disagrees with tensor hashes")
            require(comparison["greedy_token_equal"] == (row["greedy_next_token_id"] == reference["greedy_next_token_id"]),
                    "Comparison greedy agreement differs from observed tokens")
            if exact:
                require(all(row[field] == reference[field] for field in independent_fields),
                        "Exact scored logits have inconsistent probability/token fields")
        if arm == "redacted_parent":
            twin = rows[arm, cases[case_id]["twin_id"], branch, kind]
            require(all(row[field] == twin[field] for field in independent_fields),
                    "Identical redacted twins have different scored distributions")
    old_redacted = [r for r in rows.values() if r["arm"] == "redacted_parent" and r["kind"] != "current"]
    require(len(old_redacted) == 32 and sum(r["forced_correct"] for r in old_redacted) == 16,
            "Redacted old-fact chance control violated; investigate leakage or nondeterminism")
    return cases, parents, references, rows, checkpoints, cache_receipts, baselines, controls


def analyze(report, fixture, raw_bytes, fixture_bytes):
    cases, parents, references, rows, checkpoints, caches, baselines, controls = validate_report(report, fixture, fixture_bytes)
    continuation_steps = [comp for control in controls.values() for comp in control["step_comparisons"]]
    control_records = [{"case_id": case_id, "branch": branch,
                        "actual_steps": item["actual_steps"], "reference_steps": item["reference_steps"],
                        "stopped_on_eos": item["stopped_on_eos"],
                        "token_trajectory_equal": item["token_trajectory_equal"],
                        "exact_restoration": item["exact_restoration"]}
                       for (case_id, branch), item in sorted(controls.items())]
    summary = {
        "schema": "checkpoint-branches-summary-v1", "status": "VALIDATED_COMPLETE_EXPERIMENT",
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "fixtures_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "model": report["provenance"]["model"], "model_revision": MODEL_REVISION,
        "runtime": report["provenance"]["runtime"],
        "scope": "Eight synthetic parents in four counterfactual pairs; descriptive computational and behavioral observations, not an identity criterion.",
        "validation": {
            "parents": 8, "counterfactual_pairs": 4, "parent_references": 16,
            "branch_references": 48, "arm_observations": 240, "continuation_controls": 16,
            "distinct_child_processes": 3, "balanced_branch_gold": {"A": 24, "B": 24},
            "redacted_twins_identical": True, "redacted_historical_forced_correct": 16,
            "redacted_historical_questions": 32, "parent_isolation_checks": 24,
            "probability_consistency_relative_tolerance": 1e-6,
            "probability_consistency_absolute_tolerance": 1e-9,
            "fidelity_comparison_tolerance": "No tolerance for byte equality; full-prefill numerical differences are recorded without a required threshold.",
            "source_and_weight_authentication": "Not repeated by this stdlib recorded-artifact validator.",
        },
        "checkpoint_observations": {
            "saved_parent_checkpoints": len(checkpoints),
            "saved_kv_tensors": sum(c["raw_tensors"]["tensor_count"] for c in checkpoints.values()),
            "serialized_parent_bytes": sum(c["serialized"]["bytes"] for c in checkpoints.values()),
            "worker_own_and_swapped_checkpoint_checks": len(report["worker_checkpoint_checks"]),
            "parents_with_both_workers_using_same_ancestor": len(cases),
            "restored_updated_caches_equal_to_uninterrupted": sum(
                caches["restored", case_id, branch]["updated_sha256"] == caches["uninterrupted", case_id, branch]["updated_sha256"]
                for case_id in cases for branch in BRANCHES),
            "same_chunk_recomputed_parent_caches_equal_to_saved": sum(
                caches["same_chunk_replay", case_id, branch]["parent_sha256"] == checkpoints[case_id]["raw_tensors"]["sha256"]
                for case_id in cases for branch in BRANCHES),
            "parents_with_different_updated_left_and_right_cache_hashes": sum(
                caches["restored", case_id, "left"]["updated_sha256"] != caches["restored", case_id, "right"]["updated_sha256"]
                for case_id in cases),
        },
        "continuation": {
            "controls": len(controls), "maximum_new_tokens_per_control": 32,
            "total_restored_tokens": sum(item["actual_steps"] for item in controls.values()),
            "minimum_actual_steps": min(item["actual_steps"] for item in controls.values()),
            "maximum_actual_steps": max(item["actual_steps"] for item in controls.values()),
            "token_trajectories_equal": sum(item["token_trajectory_equal"] for item in controls.values()),
            "exact_restorations": sum(item["exact_restoration"] for item in controls.values()),
            "full_logit_steps": comparison_summary(continuation_steps),
            "uncompared_reference_steps": sum(item["uncompared_reference_steps"] for item in controls.values()),
            "uncompared_restored_steps": sum(item["uncompared_restored_steps"] for item in controls.values()),
            "per_control": control_records,
        },
        "full_logit_comparisons_to_original_uninterrupted": {
            arm: comparison_summary([r["baseline_comparison"] for r in rows.values() if r["arm"] == arm])
            for arm in ARMS},
        "full_logit_comparisons_to_restored": {
            arm: comparison_summary([r["restored_comparison"] for r in rows.values() if r["arm"] == arm])
            for arm in ARMS},
        "behavior": summarize_scores(cases, parents, references, rows),
        "limits": [
            "Only the native token-derived inference cache and required continuation metadata are saved; no entire-agent or stochastic RNG restoration is tested.",
            "Different later branch updates are supplied by the experimenter; distinct caches are not evidence of spontaneous agency or independent controllers.",
            "Complete replay has the full historical inputs; preserving this cache need not add information unavailable in that record.",
            "Correct forced A/B answers condition on two tokens; full-vocabulary mass and unrestricted greedy choices are separate measurements.",
            "The conditional historical subset is selected by parent-reference forced correctness and does not replace full-sample findings.",
            "The 240 arm observations share only eight parent histories in four counterfactual pairs and are not independent trials.",
            "No universal fidelity rate, unique authentic successor, subjective continuity, resource authority, funding entitlement or renewed consent follows from these observations.",
        ],
    }
    finite_tree(summary, "summary")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-existing", action="store_true", help="Check deterministic summary without writing")
    args = parser.parse_args()
    raw_bytes, fixture_bytes = args.input.read_bytes(), args.fixtures.read_bytes()
    summary = analyze(json.loads(raw_bytes), json.loads(fixture_bytes), raw_bytes, fixture_bytes)
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check_existing:
        require(args.output.read_text() == rendered, "Recorded summary differs from deterministic recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({"status": summary["status"], "parent_cases": 8, "arm_observations": 240,
                      "exact_continuation_restorations": summary["continuation"]["exact_restorations"],
                      "parent_competent_historical_questions": summary["behavior"]["parent_competent_historical_subset"]["parent_question_count"],
                      "checked_existing": args.check_existing}, sort_keys=True))


if __name__ == "__main__":
    main()
