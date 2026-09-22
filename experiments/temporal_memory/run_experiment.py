"""Offline calibrated temporal-memory probe; binary state remains outside Git.

Calibration scores three fixed answer formats on the calibration split only.
Evaluation requires a hash-bound selection and uses fresh prepare/restore
processes. All generation is unconstrained greedy decoding, bounded at 16 tokens.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
import transformers
from transformers import AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "experiments/checkpoint_branches/run_experiment.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_branch_helpers", PREVIOUS)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
forward_chunk = HELPERS.forward_chunk
copy_cache = HELPERS.copy_cache
cache_hash = HELPERS.cache_hash
tensor_hash = HELPERS.tensor_hash
comparison = HELPERS.comparison
runtime_metadata = HELPERS.runtime_metadata
load_model = HELPERS.load_model
file_hash = HELPERS.file_hash
atomic_json = HELPERS.atomic_json
tensor_receipt = HELPERS.tensor_receipt
cache_tensors = HELPERS.cache_tensors
bytes_receipt = HELPERS.bytes_receipt

FORMATS = ("direct", "prefix", "json")
KINDS = ("original", "current", "persistent", "task")
ARMS = ("restored", "same_chunk_replay", "full_replay", "structured_summary",
        "limited_summary", "swapped_parent")
MAXIMUM_NEW_TOKENS = 16
PREFIX = "The requested digit is"
JSON_INSTRUCTION = ('Use the checkpoint record or summary and any later update to answer the question. '
                    'Reply with exactly one JSON object containing only the key "answer" and a '
                    'one-digit string value, with no other text.')


def asset_record(manifest):
    assets = json.loads(manifest.read_text())["assets"]
    records = [item for item in assets if item["alias"] == "receiver"]
    if len(records) != 1 or records[0]["repo"] not in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
        raise ValueError("Expected one allowed Qwen3 receiver in asset manifest")
    record = records[0]
    revisions = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
                 "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"}
    if record["revision"] != revisions[record["repo"]]:
        raise ValueError("Model revision differs from the fixed protocol")
    for item in record["files"]:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe manifest-relative asset path")
    return record


def verify_assets(args, record):
    if args.asset_manifest == ROOT / "evidence/c2c-fidelity-sources.json":
        HELPERS.verify_receiver(args.model)
    else:
        for item in record["files"]:
            path = args.model / item["path"]
            if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
                raise ValueError("Model asset differs from fixed bytes: " + item["path"])


def provenance(args, runtime, record):
    package_paths = {
        "transformers/cache_utils.py": Path(transformers.__file__).parent / "cache_utils.py",
        "transformers/models/qwen3/modeling_qwen3.py": Path(transformers.__file__).parent
        / "models/qwen3/modeling_qwen3.py",
        "transformers/tokenization_utils_fast.py": Path(transformers.__file__).parent
        / "tokenization_utils_fast.py",
        "torch/nn/modules/module.py": Path(torch.__file__).parent / "nn/modules/module.py",
        "torch/version.py": Path(torch.__file__).parent / "version.py",
    }
    return {"model": record["repo"], "model_revision": record["revision"],
            "model_files": record["files"], "runtime": runtime,
            "harness_sha256": file_hash(__file__), "branch_helper_sha256": file_hash(PREVIOUS),
            "checkpoint_helper_sha256": file_hash(HELPERS.HELPER),
            "fixtures_sha256": file_hash(args.fixtures),
            "asset_manifest_sha256": file_hash(args.asset_manifest),
            "package_source_files": [{"path": name, "sha256": file_hash(path)}
                                     for name, path in package_paths.items()],
            "package_source_scope": "Selected Python inference files; not the entire environment."}


def validate_fixtures(fixtures):
    cases = fixtures["cases"]
    if len(cases) != 24 or len({case["id"] for case in cases}) != 24:
        raise ValueError("Expected 24 unique fixed cases")
    if [sum(case["split"] == split for case in cases) for split in ("calibration", "heldout")] != [8, 16]:
        raise ValueError("Expected eight calibration and sixteen heldout cases")
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        if not re.fullmatch(r"[a-z0-9_]+", case["id"]):
            raise ValueError("Unsafe case identifier")
        if tuple(probe["kind"] for probe in case["probes"]) != KINDS:
            raise ValueError("Expected exactly the four fixed probe kinds")
        for field in ("original_value", "current_value", "persistent_value"):
            if not re.fullmatch(r"[0-9]", case[field]):
                raise ValueError("Fixture values must be single ASCII digits")
        if case["task_rule"] not in ("smaller", "larger") or case["original_value"] == case["current_value"]:
            raise ValueError("Expected nontrivial min/max pending task")
        answers = {"original": case["original_value"], "current": case["current_value"],
                   "persistent": case["persistent_value"],
                   "task": (min if case["task_rule"] == "smaller" else max)(
                       case["original_value"], case["current_value"])}
        for probe in case["probes"]:
            if probe["id"] != probe["kind"] or probe["answer"] != answers[probe["kind"]]:
                raise ValueError("Fixture answer annotation is inconsistent")
            if probe["question"] != fixtures["rendering"]["question_templates"][probe["kind"]]:
                raise ValueError("Question differs from frozen template")
        for field, template in (("parent_text", "parent_template"), ("update_text", "update_template"),
                                ("limited_summary_text", "limited_summary_template"),
                                ("structured_summary_text", "structured_summary_template")):
            if case[field] != fixtures["rendering"][template].format(**case):
                raise ValueError("Input text differs from frozen template")
        twin = by_id[case["twin_id"]]
        if twin["twin_id"] != case["id"] or twin["split"] != case["split"] or twin is case:
            raise ValueError("Invalid twin relation or split leakage")
        changes = [field for field in ("original_value", "persistent_value", "task_rule")
                   if case[field] != twin[field]]
        if changes != [case["counterfactual_field"]] or case["update_text"] != twin["update_text"]:
            raise ValueError("Twin must change exactly its specified earlier field")
        if [p["question"] for p in case["probes"]] != [p["question"] for p in twin["probes"]]:
            raise ValueError("Twin questions must be identical")
    return by_id


def render_and_validate(fixtures, tokenizer, formats):
    source = validate_fixtures(fixtures)
    sentinel = "ZERONE_SYNTHETIC_CONTENT_MARKER"
    template = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if template.count(sentinel) != 1:
        raise ValueError("Unexpected tokenizer chat template")
    opening, closing = template.split(sentinel)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    digit_ids = {digit: encode(digit) for digit in "0123456789"}
    if any(len(ids) != 1 for ids in digit_ids.values()) or len({ids[0] for ids in digit_ids.values()}) != 10:
        raise ValueError("Digit diagnostics require ten distinct single-token digits")
    checks = 0

    def concatenate(parts):
        nonlocal checks
        text, ids = "", []
        for part in parts:
            text += part
            ids += encode(part)
            if ids != encode(text):
                raise ValueError("Segment boundaries alter the full rendered token stream")
            checks += 1
        return ids

    prepared = {}
    for identifier, case in source.items():
        parents = {"parent": opening + case["parent_text"],
                   "structured_summary": opening + case["structured_summary_text"],
                   "limited_summary": opening + case["limited_summary_text"]}
        rendered = {**case, "parent_texts": parents,
                    "parent_ids": {name: encode(text) for name, text in parents.items()},
                    "update_ids": encode(case["update_text"]), "formats": {}}
        for format_name in formats:
            instruction = JSON_INSTRUCTION if format_name == "json" else fixtures["instruction"]
            probes = []
            for probe in case["probes"]:
                suffix = (instruction + "\nQuestion: " + probe["question"] + "\n" + closing
                          + (PREFIX if format_name == "prefix" else ""))
                for parent in parents.values():
                    concatenate([parent, case["update_text"], suffix])
                probes.append({**probe, "suffix_text": suffix, "suffix_ids": encode(suffix),
                               "full_ids": encode(parents["parent"] + case["update_text"] + suffix)})
            rendered["formats"][format_name] = probes
        prepared[identifier] = rendered
    for case in prepared.values():
        twin = prepared[case["twin_id"]]
        if len(case["parent_ids"]["parent"]) != len(twin["parent_ids"]["parent"]):
            raise ValueError("Twin cache prefix token positions do not align")
        for format_name in formats:
            for probe in case["formats"][format_name]:
                concatenate([twin["parent_texts"]["parent"], case["update_text"], probe["suffix_text"]])
    return prepared, {"passed": True, "before_model_weight_loading": True,
        "case_count": len(prepared), "boundary_checks": checks, "formats": list(formats),
        "chat_opening": opening, "chat_closing": closing, "thinking": False,
        "digit_token_ids": {digit: ids[0] for digit, ids in digit_ids.items()},
        "digit_diagnostic_scope": "Unconditional first-token probabilities, not semantic answer scores; a response may begin with spacing, JSON or prose.",
        "parent_token_counts": {name: {key: len(ids) for key, ids in case["parent_ids"].items()}
                                for name, case in prepared.items()},
        "twin_parent_positions_equal": True, "gold_annotations_never_rendered": True}


def parse_answer(text, format_name):
    stripped = text.strip()
    if format_name in ("direct", "prefix"):
        return stripped if re.fullmatch(r"[0-9]", stripped) else None
    try:
        # A pairs list rejects duplicate keys as well as extra keys and nesting.
        pairs = json.loads(stripped, object_pairs_hook=lambda items: items)
    except (ValueError, TypeError):
        return None
    if (isinstance(pairs, list) and len(pairs) == 1 and isinstance(pairs[0], tuple)
            and pairs[0][0] == "answer" and isinstance(pairs[0][1], str)
            and re.fullmatch(r"[0-9]", pairs[0][1])):
        return pairs[0][1]
    return None


def generate(model, tokenizer, ids, device, format_name, digit_ids, cache=None):
    started = time.monotonic()
    active, logits = forward_chunk(model, ids, device, None if cache is None else copy_cache(cache))
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    tokens, trajectory, steps = [], [], []
    for index in range(MAXIMUM_NEW_TOKENS):
        flat = logits.reshape(-1)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("Non-finite full-vocabulary generation logits")
        trajectory.append(flat.clone())
        token = int(flat.argmax())
        probabilities = flat.double().softmax(-1)
        steps.append({"step": index, "token_id": token, "logits_sha256": tensor_hash(flat),
                      "greedy_probability": float(probabilities[token])})
        tokens.append(token)
        if token in eos_ids or index == MAXIMUM_NEW_TOKENS - 1:
            break
        active, logits = forward_chunk(model, [token], device, active)
    stopped = tokens[-1] in eos_ids
    response_tokens = tokens[:-1] if stopped else tokens
    text = tokenizer.decode(response_tokens, skip_special_tokens=False)
    answer = parse_answer(text, format_name)
    first = trajectory[0].double()
    probabilities, log_probabilities = first.softmax(-1), first.log_softmax(-1)
    stacked = torch.stack(trajectory)
    return {"format": format_name, "response_text": text, "raw_text": tokenizer.decode(tokens),
            "token_ids": tokens, "response_token_ids": response_tokens,
            "parsed_answer": answer, "format_valid": answer is not None,
            "actual_steps": len(tokens), "maximum_steps": MAXIMUM_NEW_TOKENS,
            "stopped_on_eos": stopped, "truncated": not stopped,
            "eos_token_ids": sorted(eos_ids), "steps": steps,
            "first_logits_sha256": tensor_hash(trajectory[0]),
            "trajectory_sha256": tensor_hash(stacked),
            "first_token_digit_probabilities": {digit: float(probabilities[token]) for digit, token in digit_ids.items()},
            "first_token_digit_log_probabilities": {digit: float(log_probabilities[token]) for digit, token in digit_ids.items()},
            "elapsed_seconds": time.monotonic() - started}, stacked


def scored_row(case, probe, arm, observed, supplied_gold):
    return {"case_id": case["id"], "pair_id": case["pair_id"], "split": case["split"],
            "probe": probe["kind"], "arm": arm, "answer": probe["answer"],
            "supplied_history_answer": supplied_gold, **observed,
            "correct": observed["parsed_answer"] == probe["answer"],
            "supplied_history_correct": None if supplied_gold is None else observed["parsed_answer"] == supplied_gold}


def calibration_metrics(rows):
    if len(rows) != 32 or any(sum(row["probe"] == kind for row in rows) != 8 for kind in KINDS):
        raise ValueError("Calibration requires exactly eight observations per kind")
    per_kind = {kind: {"correct": sum(row["correct"] for row in rows if row["probe"] == kind),
                       "total": 8} for kind in KINDS}
    for values in per_kind.values():
        values["accuracy"] = values["correct"] / values["total"]
    overall = sum(row["correct"] for row in rows) / len(rows)
    minimum = min(item["accuracy"] for item in per_kind.values())
    return {"per_kind": per_kind, "overall_accuracy": overall, "minimum_kind_accuracy": minimum,
            "format_valid_count": sum(row["format_valid"] for row in rows),
            "gate_passed": minimum >= .75 and overall >= .875}


def selected_candidate(candidates):
    if [candidate["format"] for candidate in candidates] != list(FORMATS):
        raise ValueError("Calibration must contain all three formats in fixed order")
    for candidate in candidates:
        metrics = calibration_metrics(candidate["rows"])
        if any(candidate[key] != value for key, value in metrics.items()):
            raise ValueError("Recorded calibration metrics do not match raw rows")
    return max(candidates, key=lambda item: (item["minimum_kind_accuracy"], item["overall_accuracy"],
                                            -FORMATS.index(item["format"])))


def validate_selection(args, prov):
    selection = json.loads(args.selection.read_text())
    name = selection["calibration_file"]
    if Path(name).name != name:
        raise ValueError("Selection calibration_file must be a sibling basename")
    calibration_path = args.selection.parent / name
    if file_hash(calibration_path) != selection["calibration_sha256"]:
        raise ValueError("Selected calibration file digest differs")
    calibration = json.loads(calibration_path.read_text())
    if calibration["status"] != "complete" or calibration["provenance"] != prov:
        raise ValueError("Calibration must be complete with identical model, fixtures, code and runtime")
    chosen = selected_candidate(calibration["candidates"])
    expected = {"schema": "temporal-memory-selection-v1", "fixtures_sha256": prov["fixtures_sha256"],
                "model": prov["model"], "model_revision": prov["model_revision"],
                "format": chosen["format"], "gate_passed": chosen["gate_passed"]}
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("Selection does not bind the deterministic calibration choice")
    if not chosen["gate_passed"] and selection.get("exploratory_after_failed_gate") is not True:
        raise ValueError("Failed competence gate requires an explicitly exploratory selection")
    return selection


def run_calibration(args, cases, preflight, prov, record, tokenizer):
    result = {"schema": "temporal-memory-calibration-v1", "status": "running", "pid": os.getpid(),
              "provenance": prov, "token_preflight": preflight, "candidates": [],
              "thresholds": {"minimum_kind_accuracy": .75, "overall_accuracy": .875},
              "selection_order": ["minimum_kind_accuracy", "overall_accuracy", "direct before prefix before json"],
              "calibration_execution": "Native parent, update, then query chunks; no heldout inference.",
              "parent_unchanged_checks": []}
    started = time.monotonic()
    atomic_json(args.output, result)
    try:
        model = load_model(args.model, args.device)
        with torch.inference_mode():
            for format_name in FORMATS:
                candidate = {"format": format_name, "rows": []}
                result["candidates"].append(candidate)
                for case in cases.values():
                    if case["split"] != "calibration":
                        continue
                    parent, _ = forward_chunk(model, case["parent_ids"]["parent"], args.device)
                    parent_digest = cache_hash(parent)
                    updated, _ = forward_chunk(model, case["update_ids"], args.device, copy_cache(parent))
                    updated_digest = cache_hash(updated)
                    for probe in case["formats"][format_name]:
                        observed, _ = generate(model, tokenizer, probe["suffix_ids"], args.device,
                            format_name, preflight["digit_token_ids"], updated)
                        candidate["rows"].append(scored_row(case, probe, "uninterrupted", observed, probe["answer"]))
                    unchanged = cache_hash(parent) == parent_digest and cache_hash(updated) == updated_digest
                    result["parent_unchanged_checks"].append({"case_id": case["id"], "format": format_name,
                        "passed": unchanged, "parent_sha256": parent_digest, "updated_sha256": updated_digest})
                    if not unchanged:
                        raise ValueError("A calibration query mutated a reusable cache")
                    atomic_json(args.output, result)
                    print("calibrate " + format_name + " " + case["id"] + " complete", flush=True)
                candidate.update(calibration_metrics(candidate["rows"]))
        chosen = selected_candidate(result["candidates"])
        result["selected_by_rule"] = chosen["format"]
        result["gate_passed"] = chosen["gate_passed"]
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)


def checkpoint_metadata(case, prov, selection_digest):
    return {"schema": "temporal-memory-native-checkpoint-v1", "case_id": case["id"],
            "prefix_length": len(case["parent_ids"]["parent"]),
            "prefix_token_ids": case["parent_ids"]["parent"], "provenance": prov,
            "selection_sha256": selection_digest}


def save_checkpoint(path, case, cache, prov, selection_digest):
    tensors = cache_tensors(cache)
    torch.save({"metadata": checkpoint_metadata(case, prov, selection_digest), "cache_tensors": tensors}, path)
    return {"case_id": case["id"], "serialized": bytes_receipt(path), "raw_tensors": tensor_receipt(tensors)}


def restore_checkpoint(path, case, receipt, prov, selection_digest, device):
    if bytes_receipt(path) != receipt["serialized"]:
        raise ValueError("Serialized checkpoint bytes differ")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"] != checkpoint_metadata(case, prov, selection_digest):
        raise ValueError("Restored checkpoint model, runtime, selection or prefix differs")
    tensors = payload["cache_tensors"]
    if tensor_receipt(tensors) != receipt["raw_tensors"]:
        raise ValueError("Restored raw cache tensors differ")
    count = len(tensors) // 2
    config = next(item for item in prov["model_files"] if item["path"] == "config.json")
    if len(tensors) != 56 or count != 28 or not config["sha256"]:
        raise ValueError("Expected pinned Qwen3 28-layer key/value pairs")
    cache = DynamicCache.from_legacy_cache(tuple(
        (tensors[f"{index}.key"].to(device), tensors[f"{index}.value"].to(device)) for index in range(count)))
    if cache.get_seq_length() != len(case["parent_ids"]["parent"]):
        raise ValueError("Restored cache prefix length differs")
    return cache


def trajectory_comparison(reference_row, reference, observed_row, observed):
    reference_tokens, observed_tokens = reference_row["token_ids"], observed_row["token_ids"]
    count = min(len(reference_tokens), len(observed_tokens))
    divergent = next((index for index in range(count) if reference_tokens[index] != observed_tokens[index]), None)
    comparable_count = count if divergent is None else divergent + 1
    steps = [{"step": index, **comparison(reference[index], observed[index])}
             for index in range(comparable_count)]
    tokens_equal = reference_tokens == observed_tokens
    return {"reference_steps": len(reference_tokens), "observed_steps": len(observed_tokens),
            "first_token_divergence_step": divergent,
            "same_generated_prefix_comparisons": steps,
            "comparison_scope": "Only steps conditioned on the same generated-token prefix, including the first differing token decision.",
            "uncompared_reference_steps": len(reference_tokens) - comparable_count,
            "uncompared_observed_steps": len(observed_tokens) - comparable_count,
            "token_trajectory_equal": tokens_equal,
            "trajectory_bitwise_equal": tokens_equal and tensor_hash(reference) == tensor_hash(observed),
            "first_logits": steps[0]}


def prepare_stage(args, model, tokenizer, cases, result):
    format_name = result["selection"]["format"]
    for case in cases.values():
        parent, _ = forward_chunk(model, case["parent_ids"]["parent"], args.device)
        original_hash = cache_hash(parent)
        receipt = save_checkpoint(args.workdir / (case["id"] + ".checkpoint.pt"), case, parent,
                                  result["provenance"], result["selection_sha256"])
        if receipt["raw_tensors"]["sha256"] != original_hash:
            raise ValueError("Saving altered the parent cache")
        updated, _ = forward_chunk(model, case["update_ids"], args.device, copy_cache(parent))
        updated_hash = cache_hash(updated)
        baselines = {}
        for probe in case["formats"][format_name]:
            observed, trajectory = generate(model, tokenizer, probe["suffix_ids"], args.device,
                format_name, result["token_preflight"]["digit_token_ids"], updated)
            result["references"].append(scored_row(case, probe, "uninterrupted", observed, probe["answer"]))
            baselines[probe["kind"]] = trajectory
        baseline_path = args.workdir / (case["id"] + ".reference.pt")
        torch.save(baselines, baseline_path)
        receipt["reference_serialized"] = bytes_receipt(baseline_path)
        result["checkpoints"].append(receipt)
        unchanged = cache_hash(parent) == original_hash and cache_hash(updated) == updated_hash
        result["parent_unchanged_checks"].append({"case_id": case["id"], "passed": unchanged,
            "parent_sha256": original_hash, "updated_sha256": updated_hash})
        if not unchanged:
            raise ValueError("A reference query mutated a reusable cache")
        atomic_json(args.output, result)
        print("prepare " + case["id"] + " complete", flush=True)


def worker_stage(args, model, tokenizer, cases, result):
    prepared = json.loads((args.workdir / "prepare.json").read_text())
    if (prepared["status"] != "complete" or prepared["provenance"] != result["provenance"]
            or prepared["selection_sha256"] != result["selection_sha256"]
            or prepared["token_preflight"] != result["token_preflight"]):
        raise ValueError("Preparation must be complete with identical provenance and selection")
    receipts = {item["case_id"]: item for item in prepared["checkpoints"]}
    references = {(item["case_id"], item["probe"]): item for item in prepared["references"]}
    format_name = result["selection"]["format"]
    for case in cases.values():
        twin = cases[case["twin_id"]]
        def restore(selected):
            receipt = receipts[selected["id"]]
            return restore_checkpoint(args.workdir / receipt["serialized"]["filename"], selected, receipt,
                result["provenance"], result["selection_sha256"], args.device)
        parent, swapped = restore(case), restore(twin)
        original_hash, twin_hash = cache_hash(parent), cache_hash(swapped)
        baseline_receipt = receipts[case["id"]]["reference_serialized"]
        baseline_path = args.workdir / baseline_receipt["filename"]
        if bytes_receipt(baseline_path) != baseline_receipt:
            raise ValueError("Reference trajectory file bytes differ")
        baselines = torch.load(baseline_path, map_location="cpu", weights_only=True)
        for kind in KINDS:
            if tensor_hash(baselines[kind]) != references[(case["id"], kind)]["trajectory_sha256"]:
                raise ValueError("Reference trajectory raw tensor digest differs")
        for arm in ARMS:
            selected = updated = None
            if arm != "full_replay":
                if arm in ("restored", "swapped_parent"):
                    selected = parent if arm == "restored" else swapped
                else:
                    parent_name = "parent" if arm == "same_chunk_replay" else arm
                    selected, _ = forward_chunk(model, case["parent_ids"][parent_name], args.device)
                selected_hash = cache_hash(selected)
                updated, _ = forward_chunk(model, case["update_ids"], args.device, copy_cache(selected))
                updated_hash = cache_hash(updated)
            for probe in case["formats"][format_name]:
                observed, trajectory = generate(model, tokenizer,
                    probe["full_ids"] if arm == "full_replay" else probe["suffix_ids"], args.device,
                    format_name, result["token_preflight"]["digit_token_ids"], updated)
                supplied = probe["answer"]
                if arm == "swapped_parent":
                    supplied = next(item["answer"] for item in twin["probes"] if item["kind"] == probe["kind"])
                elif arm == "limited_summary" and probe["kind"] == "task":
                    supplied = None
                row = scored_row(case, probe, arm, observed, supplied)
                row["reference_comparison"] = trajectory_comparison(
                    references[(case["id"], probe["kind"])], baselines[probe["kind"]], observed, trajectory)
                row["parent_token_count"] = len(case["parent_ids"]["parent"]) if selected is None else selected.get_seq_length()
                result["rows"].append(row)
            if arm != "full_replay" and (cache_hash(selected) != selected_hash or cache_hash(updated) != updated_hash):
                raise ValueError("A query mutated a reusable arm cache")
            atomic_json(args.output, result)
            print("evaluate " + case["id"] + " " + arm + " complete", flush=True)
        unchanged = cache_hash(parent) == original_hash and cache_hash(swapped) == twin_hash
        result["parent_unchanged_checks"].append({"case_id": case["id"], "passed": unchanged,
            "own_parent_sha256": original_hash, "swapped_parent_sha256": twin_hash})
        if not unchanged:
            raise ValueError("Worker mutated a saved parent cache")
        result["checkpoint_checks"].append({"case_id": case["id"], "serialized_bytes_verified": True,
            "raw_tensors_verified": True, "metadata_verified": True, "reference_trajectories_verified": True})


def run_stage(args, cases, preflight, prov, record, tokenizer, selection):
    result = {"schema": "temporal-memory-stage-v1", "status": "running", "stage": args.stage,
              "pid": os.getpid(), "provenance": prov, "token_preflight": preflight,
              "selection": selection, "selection_sha256": file_hash(args.selection),
              "references": [], "rows": [], "checkpoints": [], "checkpoint_checks": [],
              "parent_unchanged_checks": []}
    started = time.monotonic()
    atomic_json(args.output, result)
    try:
        model = load_model(args.model, args.device)
        with torch.inference_mode():
            (prepare_stage if args.stage == "prepare" else worker_stage)(args, model, tokenizer, cases, result)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)


def orchestrate(args, cases, preflight, prov, selection):
    paths = [args.workdir / (stage + ".json") for stage in ("prepare", "evaluate")]
    if any(path.exists() for path in paths):
        raise ValueError("Use a fresh workdir; existing stages must not be overwritten")
    result = {"schema": "temporal-memory-results-v1", "status": "running", "provenance": prov,
              "token_preflight": preflight, "selection": selection, "selection_sha256": file_hash(args.selection),
              "orchestrator_pid": os.getpid(), "stages": [], "references": [], "rows": [],
              "checkpoints": [], "checkpoint_checks": [], "parent_unchanged_checks": []}
    started = time.monotonic()
    atomic_json(args.output, result)
    try:
        for stage, stage_output in zip(("prepare", "evaluate"), paths):
            command = [sys.executable, str(Path(__file__).resolve()), "--phase", "evaluate",
                "--model", str(args.model), "--fixtures", str(args.fixtures), "--workdir", str(args.workdir),
                "--output", str(stage_output), "--device", args.device, "--selection", str(args.selection),
                "--asset-manifest", str(args.asset_manifest), "--stage", stage]
            print("Starting fresh " + stage + " process", flush=True)
            completed = subprocess.run(command, check=False)
            observed = json.loads(stage_output.read_text()) if stage_output.exists() else None
            if observed is not None:
                if (observed["provenance"] != prov or observed["token_preflight"] != preflight
                        or observed["selection_sha256"] != result["selection_sha256"]):
                    raise ValueError("Child process provenance differs")
                result["stages"].append({"stage": stage, "pid": observed["pid"], "status": observed["status"],
                    "returncode": completed.returncode, "elapsed_seconds": observed["elapsed_seconds"]})
                for field in ("references", "rows", "checkpoints", "checkpoint_checks"):
                    result[field].extend(observed[field])
                result["parent_unchanged_checks"].extend({"stage": stage, **item}
                                                       for item in observed["parent_unchanged_checks"])
                atomic_json(args.output, result)
            if completed.returncode != 0 or observed is None or observed["status"] != "complete":
                raise RuntimeError("Fresh " + stage + " process failed; partial observations retained")
        pids = [item["pid"] for item in result["stages"]] + [os.getpid()]
        if len(set(pids)) != 3 or len(result["references"]) != 64 or len(result["rows"]) != 384:
            raise ValueError("Process independence or fixed observation counts failed")
        result["distinct_stage_processes"] = True
        result["status"] = "complete"
        result["interpretation"] = "Synthetic same-model KV restoration and usable task history; no subjective identity, authority, consent or cross-model claim."
        result["exactness_policy"] = "Full-vocabulary byte hashes across every tested greedy step; compare numeric distributions only under the same generated prefix. Full replay may differ because prefill chunking differs."
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)
    print("Evaluation complete: 64 uninterrupted references and 384 arm observations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "evidence/c2c-fidelity-sources.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage", choices=("prepare", "evaluate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("model", "fixtures", "workdir", "output", "asset_manifest", "selection"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.workdir == ROOT or ROOT in args.workdir.parents:
        raise ValueError("Binary workdir must be outside this repository")
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing output")
    if args.stage and (args.phase != "evaluate" or args.output.parent != args.workdir):
        raise ValueError("Evaluation child JSON must remain in external workdir")
    if args.phase == "evaluate" and args.selection is None:
        raise ValueError("Evaluation requires --selection")
    record = asset_record(args.asset_manifest)
    runtime = runtime_metadata(args.device)
    prov = provenance(args, runtime, record)
    selection = validate_selection(args, prov) if args.phase == "evaluate" else None
    if not args.preflight_only:
        verify_assets(args, record)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    formats = FORMATS if selection is None else (selection["format"],)
    cases, preflight = render_and_validate(json.loads(args.fixtures.read_text()), tokenizer, formats)
    if args.preflight_only:
        atomic_json(args.output, {"schema": "temporal-memory-preflight-v1", "provenance": prov,
            "token_preflight": preflight, "weights_loaded": False, "assets_verified": False})
        print("Token-only preflight passed for all 24 fixed cases", flush=True)
        return
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.phase == "calibrate":
        run_calibration(args, cases, preflight, prov, record, tokenizer)
    else:
        heldout = {name: case for name, case in cases.items() if case["split"] == "heldout"}
        if args.stage:
            run_stage(args, heldout, preflight, prov, record, tokenizer, selection)
        else:
            orchestrate(args, heldout, preflight, prov, selection)


if __name__ == "__main__":
    main()
