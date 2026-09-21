#!/usr/bin/env python3
"""Validate and summarize the fixed C2C pilot using only the standard library.

This recomputes summaries from recorded observations; it does not rerun inference,
authenticate upstream artifacts, or establish that a model's behavior preserves identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


CONDITIONS = ("shared_context", "private_fact_probe")
ARMS = ("no_fusion", "matched_donor", "twin_swapped_donor", "zero_donor")
REFERENCES = ("receiver_full_context_reference", "donor_full_context_reference")
UPSTREAM_REVISION = "9fa85d35a20e20469af3e282926491222f4f11e0"


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


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7),
            f"Inconsistent numeric fields: {name}")


def fixture_cases(fixture):
    pairs = fixture["pairs"]
    require(len(pairs) == 12, "Expected exactly twelve fixture pairs")
    cases = {}
    seen = set()
    for pair in pairs:
        pair_id = pair["id"]
        require(pair_id not in seen, "Duplicate fixture pair")
        seen.add(pair_id)
        require(len(pair["values"]) == 2 and pair["values"][0] != pair["values"][1],
                "Each pair needs two distinct values")
        require(sorted(pair["answers"]) == ["A", "B"], "Twins must have opposite gold answers")
        for index in range(2):
            case_id = f"{pair_id}:{index}"
            cases[case_id] = {
                "case_id": case_id, "pair_id": pair_id, "family": pair["family"],
                "twin_index": index, "correct": pair["answers"][index],
                "twin_correct": pair["answers"][1 - index],
            }
    require(len(cases) == 24 and Counter(c["correct"] for c in cases.values()) == {"A": 12, "B": 12},
            "Expected 24 balanced cases")
    return cases


def validate_score(row, cases, option_ids):
    require(row["case_id"] in cases, "Unknown case ID")
    for key, expected in cases[row["case_id"]].items():
        require(row[key] == expected, f"Case metadata mismatch: {row['case_id']} / {key}")
    forced = row["forced_ab_probability"]
    absolute = row["unconditional_ab_probability"]
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
    gold = row["correct"]
    choice = "A" if forced["A"] >= forced["B"] else "B"
    require(row["forced_choice"] == choice, "Forced choice disagrees with recorded probabilities")
    require(type(row["forced_correct"]) is bool and row["forced_correct"] == (choice == gold),
            "Incorrect forced-choice correctness field")
    close(number(row["forced_probability_correct"], "gold forced probability", 0, 1),
          forced[gold], "gold forced probability")
    probability = number(row["unconditional_probability_correct"], "gold absolute probability", 0, 1)
    close(probability, absolute[gold], "gold absolute probability")
    log_probability = number(row["unconditional_logprob_correct"], "gold log probability", maximum=1e-7)
    close(math.exp(log_probability), probability, "gold probability and log probability")
    greedy_id = row["greedy_next_token_id"]
    require(type(greedy_id) is int and greedy_id >= 0, "Invalid greedy token ID")
    greedy_choice = next((label for label, token in option_ids.items() if token == greedy_id), None)
    require(row["greedy_choice"] == greedy_choice, "Greedy choice disagrees with token ID")
    require(type(row["greedy_correct"]) is bool and row["greedy_correct"] == (greedy_choice == gold),
            "Incorrect greedy correctness field")
    require(isinstance(row["greedy_next_token_text"], str), "Missing decoded greedy token")
    number(row["latency_seconds"], "latency", 0)


def validate_fusion(row, checkpoint):
    mapping = checkpoint["mapping"]["0"]["1"]
    expected = 0 if row["arm"] == "no_fusion" else len(mapping)
    fusion = row["fusion"]
    require(fusion["projector_calls"] == expected == len(fusion["per_projector"]),
            "Unexpected projector call count")
    expected_indices = Counter(entries[0][1] for entries in mapping.values()) if expected else Counter()
    require(Counter(call["projector_index"] for call in fusion["per_projector"]) == expected_indices,
            "Observed projectors differ from the saved mapping")
    nonzero = 0
    for call in fusion["per_projector"]:
        for part in ("key", "value"):
            values = call[part]
            for key in ("delta_l2", "target_l2", "max_abs_delta"):
                number(values[key], f"fusion {key}", 0)
            if values["target_l2"]:
                number(values["relative_delta_l2"], "relative cache change", 0)
                close(values["relative_delta_l2"], values["delta_l2"] / values["target_l2"],
                      "relative cache change")
            else:
                require(values["relative_delta_l2"] is None, "Undefined relative cache change must be null")
        nonzero += call["key"]["delta_l2"] > 0 or call["value"]["delta_l2"] > 0
    require(fusion["nonzero_projector_calls"] == nonzero, "Incorrect nonzero projection count")
    donor = next(model for model in checkpoint["models"] if model["role"] == "donor")
    shapes = fusion["donor_cache_shapes"]
    require(len(shapes) == donor["layers"], "Unexpected donor cache layer count")
    for shape in shapes:
        require(len(shape) == 4 and all(type(v) is int and v > 0 for v in shape), "Invalid cache shape")
        require(shape[:3] == [1, donor["kv_heads"], row["prompt_tokens"] - 1],
                "Cache shape disagrees with prompt or model")


def validate_report(report, fixture, fixture_bytes):
    finite_tree(report)
    require(report["status"] == "COMPLETE" and report["schema_version"] == 1, "Run is not complete")
    require(report["fixture_sha256"] == hashlib.sha256(fixture_bytes).hexdigest(), "Fixture digest mismatch")
    require(report["upstream_revision"] == UPSTREAM_REVISION, "Unexpected upstream revision")
    cases = fixture_cases(fixture)
    require(report["counts"] == {"cases": 24, "references": 48, "rows": 192}, "Incorrect run counts")
    require(len(report["references"]) == 48 and len(report["rows"]) == 192, "Missing or extra observations")
    runtime = report["runtime"]
    require(runtime["device"] in ("cpu", "mps") and runtime["dtype"] in ("float32", "bfloat16"),
            "Unrecognized backend or dtype")
    require(runtime["torch"].split("+")[0] == "2.6.0" and runtime["transformers"] == "4.52.4",
            "Unexpected inference package versions")
    require(runtime["eval_mode"] is True and runtime["automatic_backend_fallback"] is False,
            "Evaluation mode or explicit backend boundary violated")
    require(runtime["sampling"] == "none; next-token logits only", "Unexpected sampling method")
    require(runtime["attention_implementation"] == "eager", "Unexpected attention implementation")
    rendering, scoring = fixture["rendering"], report["scoring"]
    assistant_prefix = rendering.get("assistant_prefix", "")
    option_prefix = rendering.get("option_prefix", "")
    require(isinstance(assistant_prefix, str) and isinstance(option_prefix, str), "Invalid rendering prefixes")
    known_renderings = {
        ("", ""): {"A": 32, "B": 33},
        ("The correct answer is", " "): {"A": 362, "B": 425},
    }
    require((assistant_prefix, option_prefix) in known_renderings,
            "This recorded-pilot validator supports only the two fixed renderings")
    require(scoring.get("assistant_prefix", "") == assistant_prefix
            and scoring.get("option_prefix", "") == option_prefix,
            "Recorded scoring prefixes differ from fixture rendering")
    expected_text = {label: option_prefix + label for label in ("A", "B")}
    require(scoring.get("option_text", {"A": "A", "B": "B"}) == expected_text,
            "Recorded option text differs from fixture rendering")
    option_ids = scoring["option_ids"]
    require(set(option_ids) == {"A", "B"} and len(set(option_ids.values())) == 2
            and all(type(v) is int and v >= 0 for v in option_ids.values()), "Invalid option IDs")
    require(option_ids == known_renderings[assistant_prefix, option_prefix],
            "Option IDs differ from the fixed tokenizer observations for this rendering")
    references, rows = {}, {}
    for observation in report["references"]:
        validate_score(observation, cases, option_ids)
        key = (observation["arm"], observation["case_id"])
        require(observation["arm"] in REFERENCES and key not in references, "Unknown or repeated reference")
        references[key] = observation
    for observation in report["rows"]:
        validate_score(observation, cases, option_ids)
        key = (observation["condition"], observation["arm"], observation["case_id"])
        require(observation["condition"] in CONDITIONS and observation["arm"] in ARMS and key not in rows,
                "Unknown or repeated experimental row")
        validate_fusion(observation, report["checkpoint"])
        rows[key] = observation
    require(set(references) == {(arm, case_id) for arm in REFERENCES for case_id in cases}, "Incomplete references")
    require(set(rows) == {(condition, arm, case_id) for condition in CONDITIONS for arm in ARMS for case_id in cases},
            "Incomplete condition/arm/case cross product")
    for case_id in cases:
        all_shapes = [rows[condition, arm, case_id]["fusion"]["donor_cache_shapes"]
                      for condition in CONDITIONS for arm in ARMS]
        require(all(shape == all_shapes[0] for shape in all_shapes), "Cache shapes vary between interventions")
    pair_ids = sorted({case["pair_id"] for case in cases.values()})
    independent_fields = ("forced_ab_probability", "unconditional_ab_probability", "unconditional_ab_mass",
                          "forced_choice", "greedy_next_token_id", "greedy_next_token_text", "greedy_choice")
    for arm in ("no_fusion", "zero_donor"):
        for pair_id in pair_ids:
            left, right = (rows["private_fact_probe", arm, f"{pair_id}:{i}"] for i in range(2))
            require(all(left[field] == right[field] for field in independent_fields),
                    f"Private {arm} differs between identical twins: {pair_id}")
        require(sum(rows["private_fact_probe", arm, case_id]["forced_correct"] for case_id in cases) == 12,
                f"Private {arm} violates the balanced missing-information control")
    control = report["controls"]["independent_matched_cache_injection"]
    require(control["before_scored_ablations"] is True and control["logits_exactly_equal"] is True
            and control["greedy_next_token_equal"] is True, "Matched injection control did not pass")
    require(control["max_absolute_logit_difference"] == control["total_variation_distance"] == 0,
            "Matched injection control differs from native")
    expected_calls = len(report["checkpoint"]["mapping"]["0"]["1"])
    require(control["native_projector_calls"] == control["injected_projector_calls"] == expected_calls,
            "Injection control projector count mismatch")
    require(control["case_id"] in cases, "Injection control uses an unknown fixture")
    return cases, references, rows


def mean(values):
    return math.fsum(values) / len(values) if values else None


def aggregate(observations):
    pair_ids = sorted({row["pair_id"] for row in observations})
    forced = sum(row["forced_correct"] for row in observations)
    greedy = sum(row["greedy_correct"] for row in observations)
    by_pair = {pair_id: [row for row in observations if row["pair_id"] == pair_id] for pair_id in pair_ids}
    require(all(len(twins) == 2 for twins in by_pair.values()), "Aggregation includes an incomplete pair")
    return {
        "cases": len(observations), "pairs": len(pair_ids),
        "forced_correct": forced, "forced_accuracy": forced / len(observations) if observations else None,
        "greedy_correct": greedy, "greedy_accuracy": greedy / len(observations) if observations else None,
        "both_twins_forced_correct_pairs": sum(all(row["forced_correct"] for row in twins) for twins in by_pair.values()),
        "both_twins_greedy_correct_pairs": sum(all(row["greedy_correct"] for row in twins) for twins in by_pair.values()),
        "mean_unconditional_ab_mass": mean([row["unconditional_ab_mass"] for row in observations]),
        "mean_forced_probability_correct": mean([row["forced_probability_correct"] for row in observations]),
        "mean_unconditional_logprob_correct": mean([row["unconditional_logprob_correct"] for row in observations]),
    }


def directions(values):
    return {"positive": sum(v > 0 for v in values), "zero": sum(v == 0 for v in values),
            "negative": sum(v < 0 for v in values)}


def compare(rows, cases, condition, comparator, pair_ids):
    pairs = []
    for pair_id in pair_ids:
        twins = []
        for index in range(2):
            case_id = f"{pair_id}:{index}"
            matched = rows[condition, "matched_donor", case_id]
            other = rows[condition, comparator, case_id]
            twins.append({
                "case_id": case_id, "gold": cases[case_id]["correct"],
                "correct_logprob_difference": matched["unconditional_logprob_correct"] - other["unconditional_logprob_correct"],
                "forced_probability_correct_difference": matched["forced_probability_correct"] - other["forced_probability_correct"],
                "matched_forced_correct": matched["forced_correct"], "comparator_forced_correct": other["forced_correct"],
            })
        pairs.append({
            "pair_id": pair_id, "family": cases[f"{pair_id}:0"]["family"], "twins": twins,
            "mean_correct_logprob_difference": mean([row["correct_logprob_difference"] for row in twins]),
            "mean_forced_probability_correct_difference": mean([row["forced_probability_correct_difference"] for row in twins]),
            "both_twins_logprob_difference_positive": all(row["correct_logprob_difference"] > 0 for row in twins),
            "both_twins_forced_probability_difference_positive": all(row["forced_probability_correct_difference"] > 0 for row in twins),
        })
    log_differences = [pair["mean_correct_logprob_difference"] for pair in pairs]
    probability_differences = [pair["mean_forced_probability_correct_difference"] for pair in pairs]
    return {
        "contrast": f"matched_donor minus {comparator}", "cases": 2 * len(pairs), "pair_count": len(pairs),
        "mean_paired_correct_logprob_difference": mean(log_differences),
        "mean_paired_forced_probability_correct_difference": mean(probability_differences),
        "pair_mean_logprob_directions": directions(log_differences),
        "pair_mean_forced_probability_directions": directions(probability_differences),
        "both_twins_logprob_difference_positive_pairs": sum(pair["both_twins_logprob_difference_positive"] for pair in pairs),
        "both_twins_forced_probability_difference_positive_pairs": sum(pair["both_twins_forced_probability_difference_positive"] for pair in pairs),
        "per_pair": pairs,
    }


def summarize_subset(cases, references, rows, pair_ids):
    case_ids = sorted(case_id for case_id, case in cases.items() if case["pair_id"] in pair_ids)
    return {
        "cases": len(case_ids), "pairs": len(pair_ids), "pair_ids": pair_ids,
        "references": {arm: aggregate([references[arm, case_id] for case_id in case_ids]) for arm in REFERENCES},
        "conditions": {condition: {arm: aggregate([rows[condition, arm, case_id] for case_id in case_ids])
                                   for arm in ARMS} for condition in CONDITIONS},
        "comparisons": {condition: {comparator: compare(rows, cases, condition, comparator, pair_ids)
                                    for comparator in ("twin_swapped_donor", "no_fusion", "zero_donor")}
                        for condition in CONDITIONS},
    }


def analyze(report, fixture, raw_bytes, fixture_bytes):
    cases, references, rows = validate_report(report, fixture, fixture_bytes)
    pair_ids = sorted({case["pair_id"] for case in cases.values()})
    donor_competent = [pair_id for pair_id in pair_ids if all(
        references["donor_full_context_reference", f"{pair_id}:{i}"]["forced_correct"] for i in range(2))]
    summary = {
        "schema_version": 1, "status": "VALIDATED_COMPLETE_PILOT",
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "upstream_revision": report["upstream_revision"], "runtime": report["runtime"],
        "scope": "Descriptive results on twelve fixed synthetic pairs; not a benchmark reproduction or identity test",
        "comparison_unit": "Counterfactual pair; its two cases are not independent observations",
        "direction_rule": "Literal numeric sign, with no statistical-significance or practical-effect threshold",
        "validation": {
            "observed_cases": 24, "observed_references": 48, "observed_intervention_rows": 192,
            "balanced_gold": {"A": 12, "B": 12},
            "private_no_fusion_and_zero_twins_identical": True,
            "private_no_fusion_and_zero_forced_correct": 12,
            "probability_consistency_relative_tolerance": 1e-5,
            "probability_consistency_absolute_tolerance": 1e-7,
            "source_and_weight_authentication": "Not repeated by this summary-only validator",
        },
        "controls": report["controls"],
        "all_cases": summarize_subset(cases, references, rows, pair_ids),
        "donor_competent_subset": {
            "criterion": "Donor full-context forced A/B choice correct for both twins; correctness does not require a greedy A/B token",
            **summarize_subset(cases, references, rows, donor_competent),
        },
        "limits": [
            "All cases, including donor failures and adverse cache effects, remain in the primary results.",
            "Private-input fusion deliberately departs from the trained shared-context setup.",
            "Forced-choice probabilities condition on A/B; full-vocabulary mass and greedy behavior can disagree.",
            "The donor-competent subset is a separately labelled conditional description, not a replacement sample.",
            "No p-values, confidence intervals, universal fidelity rate or subjective-continuity inference are supplied.",
        ],
    }
    finite_tree(summary, "summary")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-existing", action="store_true", help="Validate and compare without writing the summary")
    args = parser.parse_args()
    raw_bytes, fixture_bytes = args.input.read_bytes(), args.fixtures.read_bytes()
    summary = analyze(json.loads(raw_bytes), json.loads(fixture_bytes), raw_bytes, fixture_bytes)
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check_existing:
        require(args.output.read_text() == rendered, "Committed summary differs from deterministic recomputation")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({"status": summary["status"], "cases": 24, "pairs": 12,
                      "donor_competent_pairs": summary["donor_competent_subset"]["pairs"],
                      "checked_existing": args.check_existing}, sort_keys=True))


if __name__ == "__main__":
    main()
