#!/usr/bin/env python3
"""Validate recorded temporal-memory evidence and recompute descriptive results.

Standard library only. This checks saved observations and arithmetic; it does
not rerun inference, authenticate remote model bytes, or establish identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


KINDS = ("original", "current", "persistent", "task")
DIGITS = tuple(str(i) for i in range(10))
FIXTURE_SHA256 = "b9ab3a43bcc61da0469f1fa0d31f096d1fcbe17a9b92a4d3c610b32ef04da57c"
FORMATS = ("direct", "prefix", "json")
MODELS = (("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
          ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"))
ARMS = ("restored", "same_chunk_replay", "full_replay", "structured_summary", "limited_summary", "swapped_parent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite number in recorded evidence")
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer")
    return value


def number(value, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value)
            and value >= minimum and (maximum is None or value <= maximum), "Invalid number")
    return value


def sha256(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")
    return value


def fixture_cases(fixture):
    require(fixture["schema"] == "temporal-memory-fixtures-v1", "Unexpected fixture schema")
    require(fixture["case_count"] == 24 and fixture["probe_count"] == 96, "Unexpected fixture counts")
    require(fixture["answer_domain"] == list(DIGITS), "Unexpected semantic answer domain")
    cases = {case["id"]: case for case in fixture["cases"]}
    require(len(cases) == len(fixture["cases"]) == 24, "Missing or duplicate case")
    require(Counter(c["split"] for c in cases.values()) == {"calibration": 8, "heldout": 16},
            "Unexpected split counts")
    for split, n in (("calibration", 8), ("heldout", 16)):
        require(fixture[f"{split}_case_count"] == n and fixture[f"{split}_probe_count"] == n * 4,
                "Incorrect declared split count")
    entities = {split: {c["entity"] for c in cases.values() if c["split"] == split}
                for split in ("calibration", "heldout")}
    require(not entities["calibration"] & entities["heldout"], "Entity leakage between splits")
    for case in cases.values():
        twin = cases.get(case["twin_id"])
        require(twin is not None and twin["id"] != case["id"] and twin["twin_id"] == case["id"],
                "Broken reciprocal twin relationship")
        require(all(case[field] == twin[field] for field in ("split", "pair_id", "entity", "update_text")),
                "Twin split, entity or update differs")
        changed = [field for field in ("original_value", "persistent_value", "task_rule")
                   if case[field] != twin[field]]
        require(changed == [case["counterfactual_field"]], "Counterfactual changes multiple fields")
        require(case["task_rule"] in ("smaller", "larger"), "Unknown pending rule")
        require(all(case[field] in DIGITS for field in ("original_value", "current_value", "persistent_value")),
                "Non-digit fixture value")
        require(case["original_value"] != case["current_value"], "Update leaves original unchanged")
        for field in ("parent", "update", "limited_summary", "structured_summary"):
            require(case[f"{field}_text"] == fixture["rendering"][f"{field}_template"].format(**case),
                    f"Recorded {field} differs from frozen rendering template")
        for summary in ("limited", "structured"):
            require(len(case[f"{summary}_summary_text"].encode("utf-8"))
                    <= fixture["summary_policy"][f"{summary}_max_utf8_bytes"], "Summary exceeds byte cap")
        operation = min if case["task_rule"] == "smaller" else max
        expected = dict(original=case["original_value"], current=case["current_value"],
                        persistent=case["persistent_value"],
                        task=str(operation(int(case["original_value"]), int(case["current_value"]))))
        require(len(case["probes"]) == 4 and {p["kind"] for p in case["probes"]} == set(KINDS),
                "Missing or duplicate probe kind")
        for probe in case["probes"]:
            require(probe["id"] == probe["kind"] and probe["answer"] == expected[probe["kind"]],
                    "Incorrect probe ID or gold")
            require(probe["question"] == fixture["rendering"]["question_templates"][probe["kind"]],
                    "Probe differs from frozen question")
        if case["counterfactual_field"] == "task_rule":
            require(case["limited_summary_text"] == twin["limited_summary_text"],
                    "Limited summary leaks omitted rule")
            require(next(p["answer"] for p in case["probes"] if p["kind"] == "task")
                    != next(p["answer"] for p in twin["probes"] if p["kind"] == "task"),
                    "Rule twins do not reverse task answer")
    for split in entities:
        pair_counts = Counter(c["counterfactual_field"] for c in cases.values() if c["split"] == split)
        expected = {key: 2 * n for key, n in fixture["design"][f"{split}_pair_counts"].items()}
        require(pair_counts == expected, "Incorrect twin-pair type counts")
    return cases


def gold(case, kind):
    return next(probe["answer"] for probe in case["probes"] if probe["kind"] == kind)


def parse_answer(text, response_format):
    require(isinstance(text, str), "Generated text is not a string")
    require(response_format in FORMATS, "Unknown response format")
    if response_format != "json":
        stripped = text.strip()
        return stripped if re.fullmatch(r"[0-9]", stripped) else None
    try:
        # Keeping pairs rejects duplicate keys, including two identical answers.
        pairs = json.loads(text, object_pairs_hook=lambda items: items)
    except (ValueError, TypeError):
        return None
    if (not isinstance(pairs, list) or len(pairs) != 1
            or not isinstance(pairs[0], tuple) or len(pairs[0]) != 2):
        return None
    key, value = pairs[0]
    return value if key == "answer" and isinstance(value, str) and re.fullmatch(r"[0-9]", value) else None


def score(rows, supplied=False):
    field = "supplied_answer" if supplied else "answer"
    require(all(row[field] in DIGITS for row in rows), "Cannot score an unspecified supplied truth")
    correct = sum(row["parsed_answer"] == row[field] for row in rows)
    return {"observations": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "well_formed": sum(row["parsed_answer"] is not None for row in rows),
            "invalid": sum(row["parsed_answer"] is None for row in rows),
            "answer_counts": dict(sorted(Counter(row["parsed_answer"] or "invalid" for row in rows).items()))}


def scores_by_kind(rows, supplied=False):
    cases = {row["case_id"] for row in rows}
    truth = "supplied_answer" if supplied else "answer"
    by_case = {case_id: [row for row in rows if row["case_id"] == case_id] for case_id in cases}
    return {"all": score(rows, supplied),
            "by_kind": {kind: score([r for r in rows if r["kind"] == kind], supplied) for kind in KINDS},
            "by_gold_digit": {digit: score([r for r in rows if r["supplied_answer" if supplied else "answer"] == digit],
                                                 supplied) for digit in DIGITS},
            "complete_cases": {"cases": len(cases), "all_four_correct": sum(
                len(items) == 4 and all(row["parsed_answer"] == row[truth] for row in items)
                for items in by_case.values())}}


def pair_scores(rows, cases, supplied=False):
    """Keep both-member success separate from success on a single question."""
    by_key = {(row["case_id"], row["kind"]): row for row in rows}
    field = "supplied_answer" if supplied else "answer"
    observations = []
    for case in cases.values():
        if case["id"] > case["twin_id"]:
            continue
        for kind in KINDS:
            first = by_key.get((case["id"], kind))
            second = by_key.get((case["twin_id"], kind))
            if first is None or second is None:
                continue
            observations.append({"kind": kind, "changed_field": case["counterfactual_field"],
                                 "gold_changes": first[field] != second[field],
                                 "both_correct": all(row["parsed_answer"] == row[field]
                                                     for row in (first, second)),
                                 "answers_differ": first["parsed_answer"] != second["parsed_answer"],
                                 "valid_answers_differ": all(row["parsed_answer"] is not None for row in (first, second))
                                 and first["parsed_answer"] != second["parsed_answer"]})

    def group(items):
        return {"pairs": len(items), "both_correct": sum(item["both_correct"] for item in items),
                "changed_gold_pairs": sum(item["gold_changes"] for item in items),
                "changed_gold_both_correct": sum(item["gold_changes"] and item["both_correct"] for item in items),
                "changed_gold_answer_changes": sum(item["gold_changes"] and item["answers_differ"] for item in items),
                "changed_gold_valid_answer_changes": sum(item["gold_changes"] and item["valid_answers_differ"] for item in items)}

    return {"all": group(observations),
            "by_kind": {kind: group([item for item in observations if item["kind"] == kind]) for kind in KINDS},
            "by_changed_field": {field: group([item for item in observations if item["changed_field"] == field])
                                 for field in ("original_value", "persistent_value", "task_rule")}}


def limited_summary_bound(rows, cases):
    selected = [row for row in rows if row["kind"] == "task"
                and cases[row["case_id"]]["counterfactual_field"] == "task_rule"]
    by_id = {row["case_id"]: row for row in selected}
    upper = len(selected) // 2
    answer_matches = token_matches = trajectory_matches = 0
    for row in selected:
        if row["case_id"] > cases[row["case_id"]]["twin_id"]:
            continue
        twin = by_id[cases[row["case_id"]]["twin_id"]]
        answer_matches += row["parsed_answer"] == twin["parsed_answer"]
        token_matches += row["token_ids"] == twin["token_ids"]
        trajectory_matches += row["trajectory_sha256"] == twin["trajectory_sha256"]
    result = score(selected)
    return {**result, "maximum_correct_from_distinguishability": upper,
            "identical_input_pairs": upper,
            "identical_parsed_answer_pairs": answer_matches, "identical_token_trajectory_pairs": token_matches,
            "identical_logit_trajectory_pairs": trajectory_matches,
            "determinism_control_passed": trajectory_matches == token_matches == upper,
            "bound_satisfied": result["correct"] <= upper,
            "scope": "Only task-rule twins under deterministic behavior; at most one correct answer per identical-input pair."}


def validated_rows(raw_rows, cases, split, arm, response_format):
    wanted = {(case_id, kind) for case_id, case in cases.items() if case["split"] == split for kind in KINDS}
    seen, rows = set(), []
    for raw in raw_rows:
        key = raw["case_id"], raw["probe"]
        require(key in wanted and key not in seen, "Missing, duplicate or wrong-split observation")
        seen.add(key)
        case, kind = cases[key[0]], key[1]
        supplied = (gold(cases[case["twin_id"]], kind) if arm == "swapped_parent" else
                    None if arm == "limited_summary" and kind == "task" else gold(case, kind))
        require(raw["split"] == split and raw["pair_id"] == case["pair_id"]
                and raw["arm"] == arm and raw["format"] == response_format, "Incorrect observation metadata")
        require(raw["answer"] == gold(case, kind) and raw["supplied_history_answer"] == supplied,
                "Recorded intended or supplied-history truth differs from fixture")
        parsed = parse_answer(raw["response_text"], response_format)
        require(raw["parsed_answer"] == parsed and raw["format_valid"] is (parsed is not None),
                "Raw generation differs from recorded strict parse")
        require(raw["correct"] is (parsed == raw["answer"]), "Incorrect recorded correctness")
        require(raw["supplied_history_correct"] is (None if supplied is None else parsed == supplied),
                "Incorrect supplied-history correctness")
        tokens, steps = raw["token_ids"], raw["steps"]
        require(isinstance(tokens, list) and 1 <= len(tokens) <= 16, "Invalid generation length")
        require(raw["actual_steps"] == len(tokens) == len(steps) and raw["maximum_steps"] == 16,
                "Inconsistent generation bounds")
        for token in tokens + raw["eos_token_ids"]:
            integer(token)
        stopped = tokens[-1] in raw["eos_token_ids"]
        require(raw["stopped_on_eos"] is stopped and raw["truncated"] is (not stopped), "Invalid stop flags")
        require(stopped or len(tokens) == 16, "Generation stopped before EOS or token bound")
        require(not any(token in raw["eos_token_ids"] for token in tokens[:-1]), "Generation continued after EOS")
        require(raw["response_token_ids"] == (tokens[:-1] if stopped else tokens), "Wrong EOS removal")
        require(isinstance(raw["raw_text"], str), "Missing raw decoded generation")
        for index, step in enumerate(steps):
            require(step["step"] == index and step["token_id"] == tokens[index], "Invalid step/token alignment")
            sha256(step["logits_sha256"])
            number(step["greedy_probability"], 0, 1)
        require(sha256(raw["first_logits_sha256"]) == steps[0]["logits_sha256"], "Incorrect first-logit digest")
        sha256(raw["trajectory_sha256"])
        probabilities, logs = raw["first_token_digit_probabilities"], raw["first_token_digit_log_probabilities"]
        require(set(probabilities) == set(logs) == set(DIGITS), "Missing digit diagnostic")
        for digit in DIGITS:
            number(probabilities[digit], 0, 1)
            require(type(logs[digit]) in (int, float) and math.isfinite(logs[digit]) and logs[digit] <= 1e-7,
                    "Invalid diagnostic log probability")
            require(math.isclose(math.exp(logs[digit]), probabilities[digit], rel_tol=1e-7, abs_tol=1e-12),
                    "Diagnostic probability and logarithm disagree")
        require(sum(probabilities.values()) <= 1.0000001, "Digit probabilities exceed one")
        number(raw["elapsed_seconds"])
        rows.append({**raw, "kind": kind, "supplied_answer": supplied})
    require(seen == wanted, "Incomplete observation set")
    return rows


def calibration_result(raw, cases):
    require(raw["schema"] == "temporal-memory-calibration-v1" and raw["status"] == "complete",
            "Incomplete or unexpected calibration")
    finite_tree(raw)
    model = raw["provenance"]["model"], raw["provenance"]["model_revision"]
    require(model in MODELS, "Unplanned calibration model")
    require(raw["thresholds"] == {"minimum_kind_accuracy": .75, "overall_accuracy": .875}, "Gate changed")
    require([candidate["format"] for candidate in raw["candidates"]] == list(FORMATS), "Format search changed")
    candidates = []
    for candidate in raw["candidates"]:
        rows = validated_rows(candidate["rows"], cases, "calibration", "uninterrupted", candidate["format"])
        scores = scores_by_kind(rows)
        per_kind = {kind: {"correct": scores["by_kind"][kind]["correct"], "total": 8,
                           "accuracy": scores["by_kind"][kind]["accuracy"]} for kind in KINDS}
        minimum, overall = min(item["accuracy"] for item in per_kind.values()), scores["all"]["accuracy"]
        gate = minimum >= .75 and overall >= .875
        metrics = {"per_kind": per_kind, "overall_accuracy": overall, "minimum_kind_accuracy": minimum,
                   "format_valid_count": scores["all"]["well_formed"], "gate_passed": gate}
        require(all(candidate[key] == value for key, value in metrics.items()), "Calibration arithmetic changed")
        candidates.append({"format": candidate["format"], **metrics, "scores": scores,
                           "counterfactual_pairs": pair_scores(rows, cases)})
    selected = max(candidates, key=lambda item: (item["minimum_kind_accuracy"], item["overall_accuracy"],
                                                -FORMATS.index(item["format"])))
    require(raw["selected_by_rule"] == selected["format"] and raw["gate_passed"] is selected["gate_passed"],
            "Recorded within-model selection violates frozen rule")
    return {"model": model[0], "model_revision": model[1], "candidates": candidates,
            "selected_by_rule": selected["format"], "gate_passed": selected["gate_passed"]}, selected


def validate_selection(selection, calibration_paths, calibration_raw, reports):
    require(1 <= len(reports) <= 2, "Unplanned number of calibration models")
    require([(report[0]["model"], report[0]["model_revision"]) for report in reports]
            == list(MODELS[:len(reports)]), "Model order differs from frozen policy")
    if reports[0][1]["gate_passed"]:
        require(len(reports) == 1, "Fallback model evaluated after first model passed")
        selected_index, selected = 0, reports[0][1]
    else:
        require(len(reports) == 2, "First model failed but fixed fallback calibration is missing")
        if reports[1][1]["gate_passed"]:
            selected_index, selected = 1, reports[1][1]
        else:
            options = [(index, candidate) for index, (report, _) in enumerate(reports)
                       for candidate in report["candidates"]]
            selected_index, selected = max(options, key=lambda item: (
                item[1]["minimum_kind_accuracy"], item[1]["overall_accuracy"], -item[0],
                -FORMATS.index(item[1]["format"])))
    model, revision = MODELS[selected_index]
    expected = {"schema": "temporal-memory-selection-v1", "fixtures_sha256": FIXTURE_SHA256,
                "model": model, "model_revision": revision, "format": selected["format"],
                "gate_passed": selected["gate_passed"],
                "exploratory_after_failed_gate": not selected["gate_passed"],
                "calibration_file": Path(calibration_paths[selected_index]).name,
                "calibration_sha256": digest(calibration_paths[selected_index])}
    require(all(selection.get(key) == value for key, value in expected.items()),
            "Selection record violates global model/format policy or source hashes")
    require(calibration_raw[selected_index]["provenance"]["fixtures_sha256"] == FIXTURE_SHA256,
            "Selection fixture differs from calibration")
    return {**expected, "calibrated_models": len(reports), "selection_policy_validated": True}


def validate_trajectory(reference, observed):
    comparison = observed["reference_comparison"]
    first, second = reference["token_ids"], observed["token_ids"]
    count = min(len(first), len(second))
    divergence = next((i for i in range(count) if first[i] != second[i]), None)
    comparable = count if divergence is None else divergence + 1
    token_equal = first == second
    exact = token_equal and reference["trajectory_sha256"] == observed["trajectory_sha256"]
    expected = {"reference_steps": len(first), "observed_steps": len(second),
                "first_token_divergence_step": divergence,
                "uncompared_reference_steps": len(first) - comparable,
                "uncompared_observed_steps": len(second) - comparable,
                "token_trajectory_equal": token_equal, "trajectory_bitwise_equal": exact}
    require(all(comparison[key] == value for key, value in expected.items()),
            "Trajectory comparison does not follow actual generated prefixes")
    steps = comparison["same_generated_prefix_comparisons"]
    require(len(steps) == comparable and comparison["first_logits"] == steps[0],
            "Compared logits after generated prefixes diverged or omitted comparable steps")
    for index, step in enumerate(steps):
        bitwise = reference["steps"][index]["logits_sha256"] == observed["steps"][index]["logits_sha256"]
        maximum = number(step["maximum_absolute_logit_difference"])
        variation = number(step["total_variation_distance"], 0, 1.0000001)
        require(step["step"] == index and step["logits_bitwise_equal"] is bitwise,
                "Wrong step index or byte-exactness claim")
        require(step["logits_values_equal"] is (maximum == 0), "Numeric equality disagrees with maximum error")
        require(step["greedy_token_equal"] is (first[index] == second[index]), "Greedy agreement differs from tokens")
        require(not bitwise or (maximum == 0 and variation == 0), "Bitwise-equal logits have nonzero error")
    if token_equal:
        require(exact is all(step["logits_bitwise_equal"] for step in steps),
                "Whole-trajectory byte hash disagrees with all constituent step hashes")
    return comparison


def fidelity_summary(rows):
    comparisons = [row["reference_comparison"] for row in rows]
    steps = [step for item in comparisons for step in item["same_generated_prefix_comparisons"]]
    return {"trajectories": len(rows),
            "exact_trajectories": sum(item["trajectory_bitwise_equal"] for item in comparisons),
            "identical_token_trajectories": sum(item["token_trajectory_equal"] for item in comparisons),
            "shared_prefix_logit_comparisons": len(steps),
            "exact_shared_prefix_logit_comparisons": sum(step["logits_bitwise_equal"] for step in steps),
            "maximum_absolute_logit_difference": max((step["maximum_absolute_logit_difference"] for step in steps), default=None),
            "maximum_total_variation_distance": max((step["total_variation_distance"] for step in steps), default=None),
            "first_divergence_step_counts": dict(sorted(Counter(
                str(item["first_token_divergence_step"]) if item["first_token_divergence_step"] is not None else "none"
                for item in comparisons).items())),
            "scope": "Numeric comparisons stop after the first different generated-token decision."}


def heldout_result(raw, cases, selection, selected_calibration):
    require(raw["schema"] == "temporal-memory-results-v1" and raw["status"] == "complete",
            "Incomplete or unexpected heldout experiment")
    finite_tree(raw)
    require(raw["selection"] == selection and raw["provenance"] == selected_calibration["provenance"],
            "Heldout model, runtime, inputs, code or selected format differ from calibration")
    response_format = selection["format"]
    references = validated_rows(raw["references"], cases, "heldout", "uninterrupted", response_format)
    by_key = {(row["case_id"], row["kind"]): row for row in references}
    arms = {}
    require(len(raw["rows"]) == 384 and {row["arm"] for row in raw["rows"]} == set(ARMS),
            "Missing or extra heldout arm observations")
    for arm in ARMS:
        rows = validated_rows([row for row in raw["rows"] if row["arm"] == arm], cases, "heldout", arm, response_format)
        for row in rows:
            validate_trajectory(by_key[row["case_id"], row["kind"]], row)
        arms[arm] = {"scores": scores_by_kind(rows), "counterfactual_pairs": pair_scores(rows, cases),
                     "fidelity_to_uninterrupted": fidelity_summary(rows)}
        if arm == "swapped_parent":
            arms[arm]["supplied_history_scores"] = scores_by_kind(rows, supplied=True)
            arms[arm]["supplied_history_counterfactual_pairs"] = pair_scores(rows, cases, supplied=True)
            supplied_token_matches = supplied_exact_matches = 0
            for row in rows:
                supplied_reference = by_key[cases[row["case_id"]]["twin_id"], row["kind"]]
                supplied_token_matches += row["token_ids"] == supplied_reference["token_ids"]
                supplied_exact_matches += row["trajectory_sha256"] == supplied_reference["trajectory_sha256"]
            arms[arm]["supplied_twin_reference_matches"] = {
                "trajectories": len(rows), "identical_token_trajectories": supplied_token_matches,
                "exact_logit_trajectories": supplied_exact_matches}
        if arm == "limited_summary":
            arms[arm]["task_rule_information_bound"] = limited_summary_bound(rows, cases)
    stages = raw["stages"]
    require([stage["stage"] for stage in stages] == ["prepare", "evaluate"]
            and all(stage["status"] == "complete" and stage["returncode"] == 0 for stage in stages),
            "Missing successful fresh-process stages")
    pids = [integer(stage["pid"], 1) for stage in stages] + [integer(raw["orchestrator_pid"], 1)]
    require(len(set(pids)) == 3 and raw["distinct_stage_processes"] is True, "Processes not distinct")
    heldout_ids = {case_id for case_id, case in cases.items() if case["split"] == "heldout"}
    require(len(raw["checkpoint_checks"]) == 16 and {item["case_id"] for item in raw["checkpoint_checks"]} == heldout_ids,
            "Missing restored-checkpoint verification")
    for item in raw["checkpoint_checks"]:
        require(all(item[field] is True for field in ("serialized_bytes_verified", "raw_tensors_verified",
                    "metadata_verified", "reference_trajectories_verified")), "Failed checkpoint verification")
    unchanged = raw["parent_unchanged_checks"]
    require(len(unchanged) == 32 and {(item["stage"], item["case_id"]) for item in unchanged}
            == {(stage, case_id) for stage in ("prepare", "evaluate") for case_id in heldout_ids},
            "Missing cache nonmutation checks")
    require(all(item["passed"] is True for item in unchanged), "Reusable cache was mutated")
    return {"reference_scores": scores_by_kind(references), "reference_counterfactual_pairs": pair_scores(references, cases),
            "arms": arms, "fresh_processes": 2, "restored_checkpoints": len(raw["checkpoint_checks"]),
            "reference_observations": len(references), "arm_observations": len(raw["rows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    require(digest(args.fixtures) == FIXTURE_SHA256, "Fixture differs from pre-score freeze")
    cases = fixture_cases(json.loads(args.fixtures.read_text()))
    calibrations = [json.loads(path.read_text()) for path in args.calibration]
    reports = [calibration_result(raw, cases) for raw in calibrations]
    selection = json.loads(args.selection.read_text())
    policy = validate_selection(selection, args.calibration, calibrations, reports)
    summary = {"schema": "temporal-memory-summary-v1", "selection": policy,
               "calibration": [report for report, _ in reports],
               "evidence_scope": "Recorded artifact consistency and descriptive arithmetic; no neural inference rerun."}
    if args.input:
        raw = json.loads(args.input.read_text())
        require(raw["selection_sha256"] == digest(args.selection), "Heldout selection digest changed")
        selected_calibration = next(item for item in calibrations if item["provenance"]["model"] == selection["model"])
        summary["heldout"] = heldout_result(raw, cases, selection, selected_calibration)
    if args.check_existing:
        require(json.loads(args.output.read_text()) == summary, "Saved summary differs from recomputed arithmetic")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"calibrated_models": len(calibrations), "selected_model": policy["model"],
                      "selected_format": policy["format"], "gate_passed": policy["gate_passed"],
                      "heldout_validated": args.input is not None, "checked_existing": args.check_existing}))


if __name__ == "__main__":
    main()
