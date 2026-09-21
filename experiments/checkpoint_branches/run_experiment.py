"""Bounded, offline same-model checkpoint restoration and branch experiment.

The default entry point starts three sequential fresh Python processes. Binary
checkpoints and full-vocabulary reference logits stay in --workdir, outside this
repository; the output JSON contains synthetic prompts, scores and receipts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "experiments/c2c_fidelity/checkpoint_control.py"
SPEC = importlib.util.spec_from_file_location("native_checkpoint_helper", HELPER)
HELPER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER_MODULE)
verify_receiver = HELPER_MODULE.verify_receiver
copy_cache = HELPER_MODULE.copy_cache
compare_logits = HELPER_MODULE.compare_logits

MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CONTROL_QUESTION = "Describe the original record in a short paragraph of about 50 words."
ARMS = ("restored", "same_chunk_replay", "full_replay", "swapped_parent", "redacted_parent")
PROBE_KINDS = ("original", "persistent", "current")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-write")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def cache_tensors(cache):
    return {f"{index}.{kind}": tensor.detach().cpu().clone()
            for index, pair in enumerate(cache.to_legacy_cache())
            for kind, tensor in zip(("key", "value"), pair)}


def tensor_receipt(tensors):
    items = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "sha256": tensor_hash(tensor)} for name, tensor in sorted(tensors.items())]
    digest = hashlib.sha256(json.dumps(items, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "tensor_count": len(items), "tensors": items}


def cache_hash(cache):
    return tensor_receipt(cache_tensors(cache))["sha256"]


def bytes_receipt(path):
    return {"filename": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": file_hash(path)}


def runtime_metadata(device):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Automatic MPS fallback must be disabled")
    if torch.__version__.split("+")[0] != "2.6.0" or transformers.__version__ != "4.52.4":
        raise ValueError("Use torch 2.6.0 and transformers 4.52.4")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable; no automatic fallback")
    torch.manual_seed(20260921)
    torch.set_num_threads(6)
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": transformers.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "device": device, "dtype": "float32",
            "attention": "eager", "sampling": "greedy", "threads": 6,
            "seed": 20260921, "automatic_mps_fallback": False,
            "offline": True, "remote_code": False, "implicit_credentials": False}


def provenance(fixtures_path, runtime):
    package_paths = {
        "transformers/cache_utils.py": Path(transformers.__file__).parent / "cache_utils.py",
        "transformers/models/qwen3/modeling_qwen3.py": Path(transformers.__file__).parent
        / "models/qwen3/modeling_qwen3.py",
        "transformers/tokenization_utils_fast.py": Path(transformers.__file__).parent
        / "tokenization_utils_fast.py",
        "torch/nn/modules/module.py": Path(torch.__file__).parent / "nn/modules/module.py",
        "torch/version.py": Path(torch.__file__).parent / "version.py",
    }
    manifest = ROOT / "evidence/c2c-fidelity-sources.json"
    receiver = next(asset for asset in json.loads(manifest.read_text())["assets"]
                    if asset["alias"] == "receiver")
    return {"model": MODEL_REPO, "model_revision": MODEL_REVISION, "runtime": runtime,
            "harness_sha256": file_hash(__file__), "helper_sha256": file_hash(HELPER),
            "fixtures_sha256": file_hash(fixtures_path), "asset_manifest_sha256": file_hash(manifest),
            "model_files": receiver["files"],
            "package_source_files": [{"path": name, "sha256": file_hash(path)}
                                     for name, path in package_paths.items()],
            "package_source_scope": "Selected Python inference implementation files, not the entire installed environment."}


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def render_and_validate(fixtures, tokenizer):
    """Validate all text/token boundaries before loading model weights or scoring."""
    sentinel = "ZERONE_SYNTHETIC_CONTENT_MARKER"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if template.count(sentinel) != 1:
        raise ValueError("Unexpected chat template sentinel expansion")
    opening, closing = template.split(sentinel)
    labels = {label: encode(tokenizer, label) for label in ("A", "B")}
    if any(len(ids) != 1 for ids in labels.values()) or labels["A"] == labels["B"]:
        raise ValueError("A and B must be distinct single tokens")
    cases = fixtures["cases"]
    if len(cases) != 8 or len({case["id"] for case in cases}) != 8:
        raise ValueError("Expected eight unique fixed parent cases")
    prepared = {}
    boundary_checks = 0

    def query_text(probe):
        return (fixtures["instruction"] + "\nQuestion: " + probe["question"]
                + "\nA) " + probe["choice_A"] + "\nB) " + probe["choice_B"]
                + "\nAnswer:" + closing)

    def check_segments(parts):
        nonlocal boundary_checks
        collected = []
        text = ""
        for part in parts:
            text += part
            collected += encode(tokenizer, part)
            if collected != encode(tokenizer, text):
                raise ValueError("Text split changes the concatenated token stream")
            boundary_checks += 1
        return collected

    for case in cases:
        if not case["id"].replace("_", "").isalnum():
            raise ValueError("Unsafe case identifier")
        parent_text = opening + case["parent_text"]
        redacted_text = opening + case["redacted_parent_text"]
        parent_ids, redacted_ids = encode(tokenizer, parent_text), encode(tokenizer, redacted_text)
        if len(parent_ids) != len(redacted_ids):
            raise ValueError("Redacted parent token positions differ")
        branches = {}
        if {branch["id"] for branch in case["branches"]} != {"left", "right"}:
            raise ValueError("Expected exactly the left and right branches")
        for branch in case["branches"]:
            if tuple(probe["kind"] for probe in branch["probes"]) != PROBE_KINDS:
                raise ValueError("Expected original, persistent and current probes")
            probes = []
            for probe in branch["probes"]:
                if probe["correct"] not in labels or probe["id"] != probe["kind"]:
                    raise ValueError("Invalid fixed probe metadata")
                suffix = query_text(probe)
                full_ids = check_segments([parent_text, branch["update_text"], suffix])
                check_segments([redacted_text, branch["update_text"], suffix])
                for label in labels:
                    if encode(tokenizer, parent_text + branch["update_text"] + suffix + label) != full_ids + labels[label]:
                        raise ValueError("Answer-label token changes at the actual continuation boundary")
                    boundary_checks += 1
                probes.append({**probe, "suffix_text": suffix, "suffix_ids": encode(tokenizer, suffix),
                               "full_ids": full_ids})
            branches[branch["id"]] = {"id": branch["id"], "update_text": branch["update_text"],
                                       "update_ids": encode(tokenizer, branch["update_text"]), "probes": probes}
        controls = []
        for probe in branches["left"]["probes"][:2]:
            check_segments([parent_text, probe["suffix_text"]])
            controls.append(probe)
        control_text = CONTROL_QUESTION + closing
        check_segments([parent_text, control_text])
        prepared[case["id"]] = {"id": case["id"], "twin_id": case["twin_id"],
                                 "parent_text": parent_text, "parent_ids": parent_ids,
                                 "redacted_parent_text": redacted_text, "redacted_parent_ids": redacted_ids,
                                 "branches": branches, "parent_probes": controls,
                                 "control_suffix_text": control_text,
                                 "control_suffix_ids": encode(tokenizer, control_text)}
    for case in prepared.values():
        twin = prepared[case["twin_id"]]
        if twin["twin_id"] != case["id"] or twin["id"] == case["id"]:
            raise ValueError("Twins must be reciprocal distinct cases")
        if len(case["parent_ids"]) != len(twin["parent_ids"]) or case["parent_ids"] == twin["parent_ids"]:
            raise ValueError("Twin parent positions must align while their token content differs")
        if case["redacted_parent_ids"] != twin["redacted_parent_ids"]:
            raise ValueError("Redacted twins must be identical")
        for branch_name in ("left", "right"):
            branch, twin_branch = case["branches"][branch_name], twin["branches"][branch_name]
            if branch["update_ids"] != twin_branch["update_ids"]:
                raise ValueError("Twin branch updates must be identical")
            for probe, twin_probe in zip(branch["probes"], twin_branch["probes"]):
                if probe["suffix_ids"] != twin_probe["suffix_ids"]:
                    raise ValueError("Twin probe suffixes must be identical")
                if (probe["correct"] == twin_probe["correct"]) != (probe["kind"] == "current"):
                    raise ValueError("Counterfactual gold pattern differs from the fixed design")
                check_segments([twin["parent_text"], branch["update_text"], probe["suffix_text"]])
    receipt = {"passed": True, "before_model_weight_loading": True, "case_count": len(prepared),
               "branch_question_count": 48, "boundary_checks": boundary_checks,
               "labels": {label: ids[0] for label, ids in labels.items()},
               "chat_opening": opening, "chat_closing": closing,
               "parent_token_counts": {name: len(case["parent_ids"]) for name, case in prepared.items()},
               "parent_token_ids": {name: case["parent_ids"] for name, case in prepared.items()},
               "redacted_parent_token_ids": {name: case["redacted_parent_ids"] for name, case in prepared.items()},
               "twin_and_redacted_positions_equal": True, "redacted_twins_identical": True,
               "gold_never_rendered": True, "thinking": False,
               "control_suffix_text": CONTROL_QUESTION, "control_maximum_new_tokens": 32}
    return prepared, receipt


def load_model(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager").to(device).eval()


def forward_chunk(model, token_ids, device, cache=None):
    if not token_ids:
        raise ValueError("Empty inference chunk")
    start = 0 if cache is None else cache.get_seq_length()
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    positions = torch.arange(start, start + len(token_ids), device=device)
    output = model(input_ids=ids,
                   attention_mask=torch.ones((1, start + len(token_ids)), dtype=torch.long, device=device),
                   position_ids=positions[None, :], cache_position=positions,
                   past_key_values=cache, use_cache=True, return_dict=True)
    result_cache = output.past_key_values
    logits = output.logits[:, -1, :].detach().cpu().clone()
    if result_cache.get_seq_length() != start + len(token_ids):
        raise ValueError("Unexpected cache sequence length")
    return result_cache, logits


def score(logits, probe, label_ids, tokenizer, supplied_gold):
    values = logits.detach().cpu().double().reshape(-1)
    probabilities = values.softmax(-1)
    logprobabilities = values.log_softmax(-1)
    ab_logits = torch.stack([values[label_ids[label]] for label in ("A", "B")])
    ab_probabilities = ab_logits.softmax(-1)
    chosen = ("A", "B")[int(ab_logits.argmax())]
    greedy_id = int(values.argmax())
    greedy_label = next((label for label, identifier in label_ids.items() if identifier == greedy_id), None)
    return {"correct": probe["correct"], "supplied_history_correct": supplied_gold,
            "forced_choice": chosen, "forced_correct": chosen == probe["correct"],
            "forced_supplied_history_correct": None if supplied_gold is None else chosen == supplied_gold,
            "conditional_probabilities": {label: float(ab_probabilities[index])
                                          for index, label in enumerate(("A", "B"))},
            "absolute_probabilities": {label: float(probabilities[label_ids[label]]) for label in ("A", "B")},
            "absolute_log_probabilities": {label: float(logprobabilities[label_ids[label]]) for label in ("A", "B")},
            "absolute_ab_mass": float(sum(probabilities[label_ids[label]] for label in ("A", "B"))),
            "greedy_token_id": greedy_id, "greedy_token_text": tokenizer.decode([greedy_id]),
            "greedy_label": greedy_label, "greedy_correct": greedy_label == probe["correct"],
            "greedy_supplied_history_correct": None if supplied_gold is None else greedy_label == supplied_gold,
            "logits_sha256": tensor_hash(logits)}


def comparison(reference, observed):
    return {**compare_logits(reference, observed),
            "logits_values_equal": bool(torch.equal(reference, observed)),
            "logits_bitwise_equal": tensor_hash(reference) == tensor_hash(observed)}


def continuation(model, cache, suffix_ids, device, tokenizer):
    active, logits = forward_chunk(model, suffix_ids, device, copy_cache(cache))
    tokens, trajectory = [], []
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    for step in range(32):
        trajectory.append(logits.reshape(-1))
        token = int(logits.argmax())
        tokens.append(token)
        if token in eos_ids or step == 31:
            break
        active, logits = forward_chunk(model, [token], device, active)
    return {"token_ids": tokens, "text": tokenizer.decode(tokens), "actual_steps": len(tokens),
            "maximum_steps": 32, "stopped_on_eos": tokens[-1] in eos_ids,
            "eos_token_ids": sorted(eos_ids)}, torch.stack(trajectory)


def checkpoint_metadata(case, prov):
    return {"schema": "synthetic-native-kv-checkpoint-v1", "case_id": case["id"],
            "prefix_length": len(case["parent_ids"]), "prefix_token_ids": case["parent_ids"],
            "model": MODEL_REPO, "model_revision": MODEL_REVISION,
            "runtime": prov["runtime"], "provenance": prov}


def save_checkpoint(path, case, cache, prov):
    tensors = cache_tensors(cache)
    receipt = tensor_receipt(tensors)
    payload = {"metadata": checkpoint_metadata(case, prov), "cache_tensors": tensors}
    torch.save(payload, path)
    return {"case_id": case["id"], "prefix_length": len(case["parent_ids"]),
            "serialized": bytes_receipt(path), "raw_tensors": receipt}


def restore_checkpoint(path, case, receipt, prov, device):
    if bytes_receipt(path) != receipt["serialized"]:
        raise ValueError("Serialized checkpoint byte receipt differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"] != checkpoint_metadata(case, prov):
        raise ValueError("Checkpoint runtime, model, prefix or source provenance differs")
    tensors = payload["cache_tensors"]
    if tensor_receipt(tensors) != receipt["raw_tensors"]:
        raise ValueError("Reloaded raw checkpoint tensors differ")
    count = len(tensors) // 2
    if len(tensors) != 56 or count != 28:
        raise ValueError("Expected 28 key/value layer pairs")
    cache = DynamicCache.from_legacy_cache(tuple(
        (tensors[f"{index}.key"].to(device), tensors[f"{index}.value"].to(device))
        for index in range(count)))
    if cache.get_seq_length() != len(case["parent_ids"]):
        raise ValueError("Restored checkpoint has the wrong prefix length")
    return cache


def base_stage(stage, prov, preflight):
    return {"schema": "checkpoint-branches-stage-v1", "stage": stage, "pid": os.getpid(),
            "status": "running", "provenance": prov, "token_preflight": preflight,
            "parent_references": [], "branch_references": [], "checkpoints": [],
            "continuation_baselines": [], "continuation_controls": [], "rows": [],
            "branch_cache_receipts": [], "parent_unchanged_checks": []}


def prepare_stage(args, model, tokenizer, cases, result):
    labels = result["token_preflight"]["labels"]
    for case in cases.values():
        parent, _ = forward_chunk(model, case["parent_ids"], args.device)
        original_hash = cache_hash(parent)
        receipt = save_checkpoint(args.workdir / (case["id"] + ".checkpoint.pt"), case,
                                  parent, result["provenance"])
        if receipt["raw_tensors"]["sha256"] != original_hash:
            raise ValueError("Saving altered the parent cache")
        baseline_tensors = {}
        for probe in case["parent_probes"]:
            _, logits = forward_chunk(model, probe["suffix_ids"], args.device, copy_cache(parent))
            result["parent_references"].append({"case_id": case["id"], "probe": probe["kind"],
                "arm": "parent_uninterrupted", **score(logits, probe, labels, tokenizer, probe["correct"])})
            baseline_tensors["parent." + probe["kind"]] = logits
        control, trajectory = continuation(model, parent, case["control_suffix_ids"], args.device, tokenizer)
        result["continuation_baselines"].append({"case_id": case["id"], **control,
                                                 "logits_sha256": tensor_hash(trajectory)})
        baseline_tensors["control_logits"] = trajectory
        baseline_tensors["control_tokens"] = torch.tensor(control["token_ids"], dtype=torch.long)
        for branch in case["branches"].values():
            updated, _ = forward_chunk(model, branch["update_ids"], args.device, copy_cache(parent))
            updated_hash = cache_hash(updated)
            result["branch_cache_receipts"].append({"case_id": case["id"], "branch": branch["id"],
                "arm": "uninterrupted", "parent_sha256": original_hash,
                "updated_sha256": updated_hash, "updated_length": updated.get_seq_length()})
            for probe in branch["probes"]:
                _, logits = forward_chunk(model, probe["suffix_ids"], args.device, copy_cache(updated))
                result["branch_references"].append({"case_id": case["id"], "branch": branch["id"],
                    "probe": probe["kind"], "arm": "uninterrupted",
                    **score(logits, probe, labels, tokenizer, probe["correct"])})
                baseline_tensors["branch." + branch["id"] + "." + probe["kind"]] = logits
            if cache_hash(updated) != updated_hash:
                raise ValueError("A query mutated its reusable branch cache")
        unchanged = cache_hash(parent) == original_hash
        result["parent_unchanged_checks"].append({"case_id": case["id"], "passed": unchanged})
        if not unchanged:
            raise ValueError("A continuation or branch mutated its reusable parent cache")
        baseline_path = args.workdir / (case["id"] + ".baseline.pt")
        torch.save(baseline_tensors, baseline_path)
        receipt["baseline_serialized"] = bytes_receipt(baseline_path)
        result["checkpoints"].append(receipt)
        atomic_json(args.output, result)
        print("prepare " + case["id"] + " complete", flush=True)


def worker_stage(args, model, tokenizer, cases, result):
    prepared = json.loads((args.workdir / "prepare.json").read_text())
    if prepared["status"] != "complete" or prepared["provenance"] != result["provenance"]:
        raise ValueError("Completed prepare stage with identical provenance is required")
    if prepared["token_preflight"] != result["token_preflight"]:
        raise ValueError("Worker token preflight differs from prepare")
    receipts = {item["case_id"]: item for item in prepared["checkpoints"]}
    controls = {item["case_id"]: item for item in prepared["continuation_baselines"]}
    labels = result["token_preflight"]["labels"]
    for case in cases.values():
        receipt = receipts[case["id"]]
        parent = restore_checkpoint(args.workdir / receipt["serialized"]["filename"], case,
                                    receipt, result["provenance"], args.device)
        original_hash = cache_hash(parent)
        baseline_path = args.workdir / receipt["baseline_serialized"]["filename"]
        if bytes_receipt(baseline_path) != receipt["baseline_serialized"]:
            raise ValueError("Baseline tensor bytes differ")
        baselines = torch.load(baseline_path, map_location="cpu", weights_only=True)
        if tensor_hash(baselines["control_logits"]) != controls[case["id"]]["logits_sha256"]:
            raise ValueError("Baseline control tensor digest differs")
        control, trajectory = continuation(model, parent, case["control_suffix_ids"], args.device, tokenizer)
        reference_trajectory = baselines["control_logits"]
        comparisons = [comparison(reference_trajectory[index], trajectory[index])
                       for index in range(min(len(reference_trajectory), len(trajectory)))]
        token_match = control["token_ids"] == baselines["control_tokens"].tolist()
        control_exact = (len(reference_trajectory) == len(trajectory) and token_match
                         and all(item["logits_bitwise_equal"] for item in comparisons))
        result["continuation_controls"].append({"case_id": case["id"], "branch": args.stage,
            **control, "reference_steps": len(reference_trajectory),
            "logits_sha256": tensor_hash(trajectory), "token_trajectory_equal": token_match,
            "exact_restoration": control_exact, "step_comparisons": comparisons,
            "uncompared_reference_steps": max(0, len(reference_trajectory) - len(trajectory)),
            "uncompared_restored_steps": max(0, len(trajectory) - len(reference_trajectory))})
        branch = case["branches"][args.stage]
        twin = cases[case["twin_id"]]
        twin_receipt = receipts[twin["id"]]
        swapped = restore_checkpoint(args.workdir / twin_receipt["serialized"]["filename"], twin,
                                     twin_receipt, result["provenance"], args.device)
        twin_hash = cache_hash(swapped)
        replayed, _ = forward_chunk(model, case["parent_ids"], args.device)
        redacted, _ = forward_chunk(model, case["redacted_parent_ids"], args.device)
        arm_parents = {"restored": parent, "same_chunk_replay": replayed,
                       "swapped_parent": swapped, "redacted_parent": redacted}
        restored_logits = {}
        for arm in ARMS:
            started = time.monotonic()
            updated = None
            if arm != "full_replay":
                selected = arm_parents[arm]
                selected_hash = cache_hash(selected)
                updated, _ = forward_chunk(model, branch["update_ids"], args.device, copy_cache(selected))
                updated_hash = cache_hash(updated)
                result["branch_cache_receipts"].append({"case_id": case["id"], "branch": args.stage,
                    "arm": arm, "parent_sha256": selected_hash, "updated_sha256": updated_hash,
                    "updated_length": updated.get_seq_length()})
            for probe in branch["probes"]:
                row_started = time.monotonic()
                if arm == "full_replay":
                    _, logits = forward_chunk(model, probe["full_ids"], args.device)
                else:
                    _, logits = forward_chunk(model, probe["suffix_ids"], args.device, copy_cache(updated))
                supplied_gold = probe["correct"]
                if arm == "swapped_parent" and probe["kind"] != "current":
                    supplied_gold = next(item["correct"] for item in twin["branches"][args.stage]["probes"]
                                         if item["kind"] == probe["kind"])
                if arm == "redacted_parent" and probe["kind"] != "current":
                    supplied_gold = None
                reference = baselines["branch." + args.stage + "." + probe["kind"]]
                if arm == "restored":
                    restored_logits[probe["kind"]] = logits.clone()
                result["rows"].append({"case_id": case["id"], "branch": args.stage,
                    "probe": probe["kind"], "arm": arm,
                    **score(logits, probe, labels, tokenizer, supplied_gold),
                    "baseline_comparison": comparison(reference, logits),
                    "restored_comparison": comparison(restored_logits[probe["kind"]], logits),
                    "query_elapsed_seconds": time.monotonic() - row_started})
            if arm != "full_replay" and (cache_hash(updated) != updated_hash
                                          or cache_hash(selected) != selected_hash):
                raise ValueError("An arm mutated a reusable cache")
            print(args.stage + " " + case["id"] + " " + arm + " complete in "
                  + str(round(time.monotonic() - started, 3)) + "s", flush=True)
            atomic_json(args.output, result)
        unchanged = cache_hash(parent) == original_hash and cache_hash(swapped) == twin_hash
        result["parent_unchanged_checks"].append({"case_id": case["id"], "passed": unchanged,
                                                  "own_and_swapped_checked": True})
        if not unchanged:
            raise ValueError("Worker mutated an original or swapped parent cache")
        result["checkpoints"].append({"case_id": case["id"], "serialized_bytes_verified": True,
            "raw_tensors_verified": True, "runtime_metadata_verified": True,
            "own_parent_sha256": original_hash, "swapped_parent_sha256": twin_hash})
        atomic_json(args.output, result)


def run_stage(args, cases, preflight, prov, tokenizer):
    result = base_stage(args.stage, prov, preflight)
    started = time.monotonic()
    atomic_json(args.output, result)
    try:
        verify_receiver(args.model)
        model = load_model(args.model, args.device)
        with torch.inference_mode():
            if args.stage == "prepare":
                prepare_stage(args, model, tokenizer, cases, result)
            else:
                worker_stage(args, model, tokenizer, cases, result)
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)


def orchestrate(args, preflight, prov):
    expected_paths = [args.workdir / (stage + ".json") for stage in ("prepare", "left", "right")]
    if any(path.exists() for path in expected_paths):
        raise ValueError("Use a fresh workdir; refusing to overwrite an existing experiment")
    result = {"schema": "checkpoint-branches-results-v1", "status": "running",
              "provenance": prov, "token_preflight": preflight, "orchestrator_pid": os.getpid(),
              "stages": [], "parent_references": [], "branch_references": [], "checkpoints": [],
              "continuation_baselines": [], "continuation_controls": [], "rows": [],
              "branch_cache_receipts": [], "parent_unchanged_checks": []}
    started = time.monotonic()
    atomic_json(args.output, result)
    try:
        for stage, stage_output in zip(("prepare", "left", "right"), expected_paths):
            command = [sys.executable, str(Path(__file__).resolve()), "--model", str(args.model),
                       "--fixtures", str(args.fixtures), "--workdir", str(args.workdir),
                       "--output", str(stage_output), "--device", args.device, "--stage", stage]
            print("Starting independent " + stage + " process", flush=True)
            completed = subprocess.run(command, check=False)
            observed = json.loads(stage_output.read_text()) if stage_output.exists() else None
            if observed is not None:
                if observed["provenance"] != prov or observed["token_preflight"] != preflight:
                    raise ValueError("Subprocess provenance or preflight differs")
                result["stages"].append({"stage": stage, "pid": observed["pid"],
                    "status": observed["status"], "returncode": completed.returncode,
                    "elapsed_seconds": observed["elapsed_seconds"]})
                for field in ("parent_references", "branch_references", "continuation_baselines",
                              "continuation_controls", "rows", "branch_cache_receipts"):
                    result[field].extend(observed[field])
                result["parent_unchanged_checks"].extend({"stage": stage, **item}
                                                        for item in observed["parent_unchanged_checks"])
                if stage == "prepare":
                    result["checkpoints"] = observed["checkpoints"]
                else:
                    result.setdefault("worker_checkpoint_checks", []).extend({"stage": stage, **item}
                                                                           for item in observed["checkpoints"])
                atomic_json(args.output, result)
            if completed.returncode != 0 or observed is None or observed["status"] != "complete":
                raise RuntimeError("Independent " + stage + " process failed; partial observations retained")
        pids = [item["pid"] for item in result["stages"]]
        if len(set(pids + [os.getpid()])) != 4:
            raise ValueError("Expected three distinct child processes and one orchestrator")
        if (len(result["rows"]) != 240 or len(result["branch_references"]) != 48
                or len(result["parent_references"]) != 16 or len(result["continuation_controls"]) != 16):
            raise ValueError("Experiment observation counts differ from fixed design")
        result["distinct_stage_processes"] = True
        result["status"] = "complete"
        result["interpretation"] = ("Synthetic native inference cache restoration and externally supplied branch updates "
                                    "under fixed weights and runtime; no cross-model, subjective identity, authority or consent claim.")
        result["exactness_policy"] = ("Compare all tested full-vocabulary logits without tolerance for matched chunking. "
                                     "Report full-replay chunking differences without requiring bitwise equality. "
                                     "Generation is greedy, bounded at 32 new tokens and stops at EOS; no RNG-state claim.")
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)
    print("Experiment complete: 240 arm observations, 64 references and 16 continuation controls", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--preflight-only", action="store_true", help="Validate fixtures and token boundaries without loading weights")
    parser.add_argument("--stage", choices=("prepare", "left", "right"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.model, args.fixtures = args.model.resolve(), args.fixtures.resolve()
    args.workdir, args.output = args.workdir.resolve(), args.output.resolve()
    if args.workdir == ROOT or ROOT in args.workdir.parents:
        raise ValueError("Binary workdir must be outside this repository")
    if args.stage and args.output.parent != args.workdir:
        raise ValueError("Child stage JSON must remain inside the external workdir")
    runtime = runtime_metadata(args.device)
    fixtures = json.loads(args.fixtures.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    cases, preflight = render_and_validate(fixtures, tokenizer)
    prov = provenance(args.fixtures, runtime)
    if args.preflight_only:
        atomic_json(args.output, {"schema": "checkpoint-branches-preflight-v1", "provenance": prov,
                                  "token_preflight": preflight, "weights_loaded": False})
        print("Token-only preflight passed for all eight parent cases", flush=True)
        return
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.stage:
        run_stage(args, cases, preflight, prov, tokenizer)
    else:
        orchestrate(args, preflight, prov)


if __name__ == "__main__":
    main()
