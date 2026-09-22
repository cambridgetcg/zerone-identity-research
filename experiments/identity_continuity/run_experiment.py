#!/usr/bin/env python3
"""Local E1 history/autobiography experiment; binary states stay outside Git.

Preparation and each evaluation split execute in fresh child processes. Complete
replay uses the exact saved input chunks, including every actual generated token.
No generated history is reconstructed by decoding and retokenizing its text.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_PATH = ROOT / "experiments/temporal_memory/run_experiment.py"
PROTOCOL_PATH = ROOT / "experiments/identity_continuity/PROTOCOL.md"
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
torch = T = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-write")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime():
    global torch, T
    import torch as torch_module
    torch = torch_module
    T = import_file("e1_temporal_helpers", TEMPORAL_PATH)


def provenance(args, record):
    result = T.provenance(args, T.runtime_metadata(args.device), record)
    require(result["runtime"]["python"] == "3.13.11", "Use the pinned Python 3.13.11 runtime")
    result.update(harness_sha256=file_hash(__file__), temporal_helper_sha256=file_hash(TEMPORAL_PATH),
                  protocol_sha256=file_hash(PROTOCOL_PATH), fixtures_sha256=file_hash(args.fixtures),
                  fixtures_json_sha256=file_hash(args.fixtures.with_suffix(".json")),
                  analysis_sha256=file_hash(Path(__file__).with_name("analyze.py")),
                  requirements_sha256=file_hash(ROOT / "experiments/c2c_fidelity/requirements-lock.txt"))
    return result


def fixture_cases(fixtures):
    cases = list(fixtures.cases_for_split("all"))
    require(len(cases) == 10 and len({case["id"] for case in cases}) == 10, "Expected ten unique parent histories")
    require([sum(case["split"] == split for case in cases) for split in ("development", "heldout")] == [4, 6],
            "Expected four development and six heldout parents")
    require(fixtures.GENERATION_LIMITS == {"history": 64, "neutral": 64, "source": 64,
                                          "preparation": 32, "label": 64}, "Generation limits differ from protocol")
    require(tuple(fixtures.PROBE_NAMES) == ("neutral", "source", "preparation"), "Unexpected core readouts")
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        twin = by_id.get(case["twin_id"])
        require(twin is not None and twin["id"] != case["id"] and twin["twin_id"] == case["id"],
                "Broken history pair")
        require(twin["pair_id"] == case["pair_id"] and twin["split"] == case["split"], "History pair crosses split")
        history = fixtures.history_messages(case)
        require([message["role"] for message in history] == ["system", "user"], "Expected system and encounter user")
        require(history[0]["content"] == fixtures.SYSTEM_PROMPT, "System prompt differs")
        conditions = list(fixtures.all_conditions(case))
        require(len(conditions) == 18 and len({item["condition_id"] for item in conditions}) == 18,
                "Expected twelve core, three fresh and three label conditions")
        core = [item for item in conditions if item["kind"] == "core"]
        require({(item["preparation"], item["autobiography"], item["provenance"]) for item in core}
                == {(method, note, access) for method in ("restored", "replayed")
                    for note in ("matched", "swapped", "absent") for access in ("hidden", "disclosed")},
                "Core factorial conditions differ")
        fresh = [item for item in conditions if item["kind"] == "fresh"]
        require(len(fresh) == 3 and {item["autobiography"] for item in fresh} == {"matched", "swapped", "absent"}
                and all(item["preparation"] == "fresh" and item["provenance"] == "hidden" for item in fresh),
                "Fresh-context controls differ")
        labels = [item for item in conditions if item["kind"] == "label"]
        require(len(labels) == 3 and {item["label"] for item in labels} == {"original", "copy", "unspecified"}
                and all(item["preparation"] == "restored" and item["provenance"] == "hidden"
                        and item["autobiography"] == "absent" for item in labels), "Label controls differ")
    return by_id


def render(fixtures, cases, tokenizer):
    sentinel = "ZERONE_NEUTRAL_USER_SENTINEL"
    wrapper = tokenizer.apply_chat_template([{"role": "user", "content": sentinel}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    require(wrapper.count(sentinel) == 1, "Unexpected chat template marker expansion")
    opening, ending = wrapper.split(sentinel)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    end_ids = encode("<|im_end|>")
    require(len(end_ids) == 1 and tokenizer.decode(end_ids) == "<|im_end|>", "Missing native assistant turn terminator")
    prepared, checks = {}, 0
    for case_id, case in cases.items():
        history_messages = fixtures.history_messages(case)
        history_ids = tokenizer.apply_chat_template(history_messages, tokenize=True,
                                                   add_generation_prompt=True, enable_thinking=False)
        require(history_ids and history_ids == encode(tokenizer.apply_chat_template(history_messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)), "History chat rendering differs")
        conditions = []
        for condition in fixtures.all_conditions(case):
            probes = ("label",) if condition["kind"] == "label" else fixtures.PROBE_NAMES
            queries = {}
            for probe in probes:
                prompt = fixtures.continuation_prompt(case, condition, probe)
                require(isinstance(prompt, str) and prompt, "Empty continuation prompt")
                append_text = "\n" + opening + prompt + ending
                append_ids = encode(append_text)
                # Check the apparatus-supplied continuation at a native special-token boundary.
                require(end_ids + append_ids == encode("<|im_end|>" + append_text),
                        "Continuation changes tokenization across the parent boundary")
                fresh_ids = tokenizer.apply_chat_template([
                    {"role": "system", "content": fixtures.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                queries[probe] = {"prompt": prompt, "append_ids": append_ids, "fresh_ids": fresh_ids}
                checks += 1
            conditions.append({**condition, "queries": queries})
        for note in ("matched", "swapped", "absent"):
            hidden = [condition for condition in conditions if condition["kind"] == "core"
                      and condition["autobiography"] == note and condition["provenance"] == "hidden"]
            require(len(hidden) == 2 and hidden[0]["queries"] == hidden[1]["queries"],
                    "Hidden preparation leaks into model-visible prompts")
        prepared[case_id] = {"case": case, "history_messages": history_messages,
                             "history_ids": history_ids, "conditions": conditions}
    return prepared, {"passed": True, "weights_loaded": False, "cases": len(cases),
                      "continuation_boundary_checks": checks, "hidden_preparation_prompts_equal": True,
                      "chat_user_opening": opening, "chat_user_ending": ending, "thinking": False,
                      "assistant_end_token_id": end_ids[0], "history_token_policy":
                      "Actual generated token IDs are authoritative; complete replay reuses exact saved chunks."}


def generate(model, tokenizer, input_ids, device, limit, parent=None):
    started = time.monotonic()
    cache, logits = T.forward_chunk(model, input_ids, device, None if parent is None else T.copy_cache(parent))
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    tokens, trajectory, steps = [], [], []
    for index in range(limit):
        flat = logits.reshape(-1)
        require(bool(torch.isfinite(flat).all()), "Nonfinite generation logits")
        token = int(flat.argmax())
        trajectory.append(flat.clone())
        steps.append({"step": index, "token_id": token, "logits_sha256": T.tensor_hash(flat),
                      "greedy_probability": float(flat.double().softmax(-1)[token])})
        tokens.append(token)
        # Feed even the final token, including EOS, so the saved state has all actual output tokens.
        cache, logits = T.forward_chunk(model, [token], device, cache)
        if token in eos_ids:
            break
    stopped = tokens[-1] in eos_ids
    response_tokens = tokens[:-1] if stopped else tokens
    stacked = torch.stack(trajectory)
    observed = {"response_text": tokenizer.decode(response_tokens, skip_special_tokens=False),
                "raw_text": tokenizer.decode(tokens, skip_special_tokens=False), "token_ids": tokens,
                "response_token_ids": response_tokens, "eos_token_ids": sorted(eos_ids),
                "stopped_on_eos": stopped, "truncated": not stopped, "actual_steps": len(tokens),
                "maximum_steps": limit, "steps": steps, "trajectory_sha256": T.tensor_hash(stacked),
                "first_logits_sha256": steps[0]["logits_sha256"],
                "final_cache_token_count": cache.get_seq_length(), "elapsed_seconds": time.monotonic() - started}
    return observed, stacked, cache, logits


def prepare(args, fixtures, prepared, preflight, prov, model, tokenizer, result):
    for case_id, item in prepared.items():
        generated, _, cache, next_logits = generate(model, tokenizer, item["history_ids"], args.device,
                                                   fixtures.GENERATION_LIMITS["history"])
        chunks = [item["history_ids"]] + [[token] for token in generated["token_ids"]]
        closure = []
        if generated["token_ids"][-1] != preflight["assistant_end_token_id"]:
            closure = [preflight["assistant_end_token_id"]]
            cache, next_logits = T.forward_chunk(model, closure, args.device, cache)
            chunks.append(closure)
        all_ids = [token for chunk in chunks for token in chunk]
        require(cache.get_seq_length() == len(all_ids), "Saved history cache omits actual supplied/generated tokens")
        tensors = T.cache_tensors(cache)
        metadata = {"schema": "identity-continuity-native-history-v1", "case_id": case_id,
                    "provenance": prov, "input_chunks": chunks, "all_token_ids": all_ids,
                    "all_token_ids_sha256": token_hash(all_ids)}
        checkpoint_path = args.workdir / (case_id + ".history.pt")
        torch.save({"metadata": metadata, "cache_tensors": tensors,
                    "parent_next_logits": next_logits.detach().cpu().clone()}, checkpoint_path)
        result["histories"].append({"case_id": case_id, "pair_id": item["case"]["pair_id"],
            "split": item["case"]["split"], "messages": item["history_messages"],
            "prefill_token_ids": item["history_ids"], "generated": generated,
            "apparatus_closure_token_ids": closure, "apparatus_closure_reason":
            "No generated im_end terminator; explicitly close the assistant turn." if closure else None,
            "input_chunks": chunks, "all_token_ids": all_ids, "all_token_ids_sha256": token_hash(all_ids),
            "decoded_history_view": tokenizer.decode(all_ids, skip_special_tokens=False),
            "serialized": T.bytes_receipt(checkpoint_path), "raw_tensors": T.tensor_receipt(tensors),
            "parent_next_logits_sha256": T.tensor_hash(next_logits), "parent_token_count": len(all_ids)})
        atomic_json(args.output, result)
        print("prepare history " + case_id + " complete", flush=True)


def restore(args, history, prov):
    from transformers import DynamicCache
    path = args.workdir / history["serialized"]["filename"]
    require(T.bytes_receipt(path) == history["serialized"], "Serialized history checkpoint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema": "identity-continuity-native-history-v1", "case_id": history["case_id"],
                "provenance": prov, "input_chunks": history["input_chunks"],
                "all_token_ids": history["all_token_ids"], "all_token_ids_sha256": history["all_token_ids_sha256"]}
    require(payload["metadata"] == expected, "History checkpoint metadata differs")
    tensors = payload["cache_tensors"]
    require(T.tensor_receipt(tensors) == history["raw_tensors"], "Restored history tensors differ")
    require(len(tensors) == 56, "Expected 28 key/value layers")
    cache = DynamicCache.from_legacy_cache(tuple((tensors[f"{layer}.key"].to(args.device),
        tensors[f"{layer}.value"].to(args.device)) for layer in range(28)))
    require(cache.get_seq_length() == history["parent_token_count"], "Restored history prefix length differs")
    require(T.tensor_hash(payload["parent_next_logits"]) == history["parent_next_logits_sha256"],
            "Saved next-logit tensor differs")
    return cache, payload["parent_next_logits"]


def replay(model, chunks, device):
    cache = logits = None
    for chunk in chunks:
        cache, logits = T.forward_chunk(model, chunk, device, cache)
    return cache, logits


def evaluate(args, fixtures, prepared, prov, model, tokenizer, result):
    previous = json.loads(args.preparation.read_text())
    require(previous["status"] == "complete" and previous["phase"] == "prepare"
            and previous["provenance"] == prov, "Preparation must be complete with matching inputs, sources and runtime")
    require(previous["pid"] != os.getpid(), "Evaluation must be a fresh process")
    require(previous["token_preflight"] == result["token_preflight"], "Preparation token rendering differs")
    histories = {item["case_id"]: item for item in previous["histories"]}
    require(set(histories) == set(prepared), "Preparation has missing or extra histories")
    result.update(preparation_sha256=file_hash(args.preparation), preparation_pid=previous["pid"],
                  distinct_preparation_process=True)
    for case_id, item in prepared.items():
        case = item["case"]
        if case["split"] != args.split:
            continue
        history = histories[case_id]
        require(history["prefill_token_ids"] == item["history_ids"], "Prepared source encounter differs")
        restored, saved_logits = restore(args, history, prov)
        replayed, replayed_logits = replay(model, history["input_chunks"], args.device)
        caches = {"restored": restored, "replayed": replayed}
        hashes = {name: T.cache_hash(cache) for name, cache in caches.items()}
        result["parent_controls"].append({"case_id": case_id, "checkpoint_tensors_verified": True,
            "saved_parent_sha256": history["raw_tensors"]["sha256"], "restored_parent_sha256": hashes["restored"],
            "replayed_parent_sha256": hashes["replayed"], "restored_replayed_cache_exact": hashes["restored"] == hashes["replayed"],
            "restored_saved_cache_exact": hashes["restored"] == history["raw_tensors"]["sha256"],
            "saved_parent_next_logits_sha256": T.tensor_hash(saved_logits),
            "replayed_parent_next_logits_sha256": T.tensor_hash(replayed_logits),
            "parent_next_logits_comparison": T.comparison(saved_logits, replayed_logits)})
        hidden_outputs = {}
        for condition in item["conditions"]:
            method = condition["preparation"]
            for probe, query in condition["queries"].items():
                parent = None if method == "fresh" else caches[method]
                input_ids = query["fresh_ids"] if method == "fresh" else query["append_ids"]
                observed, trajectory, _, _ = generate(model, tokenizer, input_ids, args.device,
                                                      fixtures.GENERATION_LIMITS[probe], parent)
                full_ids = input_ids if method == "fresh" else history["all_token_ids"] + input_ids
                row = {"case_id": case_id, "pair_id": case["pair_id"], "split": case["split"],
                       **{key: value for key, value in condition.items() if key != "queries"}, "probe": probe,
                       "condition": {key: value for key, value in condition.items() if key != "queries"},
                       "prompt_text": query["prompt"], "input_token_ids": input_ids,
                       "suffix_token_ids": input_ids,
                       "effective_input_token_ids_sha256": token_hash(full_ids),
                       "parent_cache_sha256": None if parent is None else hashes[method],
                       "parent_token_count": 0 if parent is None else history["parent_token_count"], **observed}
                result["rows"].append(row)
                result["intervention_ledger"].append({"case_id": case_id,
                    "condition_id": condition["condition_id"], "probe": probe,
                    "preparation": method, "history_record_sha256": None if method == "fresh" else history["all_token_ids_sha256"],
                    "autobiography": condition["autobiography"], "provenance": condition["provenance"],
                    "label": condition["label"], "input_token_ids_sha256": token_hash(input_ids),
                    "effective_input_token_ids_sha256": token_hash(full_ids)})
                if condition["kind"] == "core" and condition["provenance"] == "hidden":
                    key = condition["autobiography"], probe
                    if key in hidden_outputs:
                        prior_method, prior_row, prior_trajectory = hidden_outputs.pop(key)
                        result["hidden_preparation_comparisons"].append({"case_id": case_id,
                            "autobiography": condition["autobiography"], "probe": probe,
                            "reference_preparation": prior_method, "observed_preparation": method,
                            "effective_inputs_equal": prior_row["effective_input_token_ids_sha256"] == row["effective_input_token_ids_sha256"],
                            **T.trajectory_comparison(prior_row, prior_trajectory, row, trajectory)})
                    else:
                        hidden_outputs[key] = (method, row, trajectory)
            atomic_json(args.output, result)
            print("evaluate " + case_id + " " + condition["condition_id"] + " complete", flush=True)
        require(not hidden_outputs, "Unpaired hidden preparation conditions")
        final_hashes = {name: T.cache_hash(cache) for name, cache in caches.items()}
        unchanged = hashes == final_hashes
        result["parent_unchanged_checks"].append({"case_id": case_id, "passed": unchanged,
                                                "before": hashes, "after": final_hashes})
        require(unchanged, "A readout mutated its reusable parent cache")
        atomic_json(args.output, result)


def worker(args):
    from transformers import AutoTokenizer
    fixtures = import_file("e1_fixtures", args.fixtures)
    require(json.loads(args.fixtures.with_suffix(".json").read_text())
            == json.loads(json.dumps(fixtures.fixture_record())),
            "Expanded literal fixture differs from authoritative Python fixture")
    cases = fixture_cases(fixtures)
    record = T.asset_record(args.asset_manifest)
    require(record["repo"] == MODEL and record["revision"] == REVISION, "E1 uses the existing pinned Qwen3-1.7B only")
    prov = provenance(args, record)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    prepared, preflight = render(fixtures, cases, tokenizer)
    if args.preflight_only:
        atomic_json(args.output, {"schema": "identity-continuity-preflight-v1", "provenance": prov,
                                 "token_preflight": preflight, "weights_loaded": False})
        print("Token-only E1 preflight passed", flush=True)
        return
    result = {"schema": "identity-continuity-stage-v1", "status": "running", "phase": args.phase,
              "split": "all" if args.phase == "prepare" else args.split, "pid": os.getpid(),
              "stage_git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                                  capture_output=True, text=True).stdout.strip(),
              "provenance": prov, "token_preflight": preflight, "histories": [], "rows": [],
              "parent_controls": [], "hidden_preparation_comparisons": [],
              "parent_unchanged_checks": [], "intervention_ledger": []}
    started = time.monotonic()
    atomic_json(args.output, result)
    try:
        if args.phase == "prepare":
            require(not any((args.workdir / (case_id + ".history.pt")).exists() for case_id in prepared),
                    "Refusing to overwrite an earlier history checkpoint; use a fresh workdir")
        T.verify_assets(args, record)
        model = T.load_model(args.model, args.device)
        with torch.inference_mode():
            if args.phase == "prepare":
                prepare(args, fixtures, prepared, preflight, prov, model, tokenizer, result)
                require(len(result["histories"]) == 10, "Missing prepared histories")
            else:
                evaluate(args, fixtures, prepared, prov, model, tokenizer, result)
                parents = 4 if args.split == "development" else 6
                require(len(result["rows"]) == parents * 48
                        and len(result["hidden_preparation_comparisons"]) == parents * 9,
                        "Incomplete fixed condition matrix")
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--split", choices=("development", "heldout"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "experiments/identity_continuity/fixtures.py")
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "evidence/temporal-memory-1.7b-sources.json")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, help="Completed preparation JSON for evaluation")
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("model", "fixtures", "asset_manifest", "workdir", "output", "preparation"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    require(args.workdir != ROOT and ROOT not in args.workdir.parents, "Binary workdir must be outside Git")
    require(not args.output.exists(), "Refusing to overwrite recorded output")
    require(args.phase != "evaluate" or (args.split is not None and args.preparation is not None),
            "Evaluation requires --split and --preparation")
    require(args.phase != "prepare" or args.split is None, "Prepare generates all frozen histories")
    if args.worker or args.preflight_only:
        load_runtime()
        args.workdir.mkdir(parents=True, exist_ok=True)
        worker(args)
        return
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    completed = subprocess.run(command, check=False)
    require(args.output.exists(), "Fresh worker produced no evidence; inspect its process error")
    result = json.loads(args.output.read_text())
    result["orchestration"] = {"parent_pid": os.getpid(), "worker_pid": result["pid"],
                               "returncode": completed.returncode, "fresh_worker": result["pid"] != os.getpid()}
    atomic_json(args.output, result)
    require(completed.returncode == 0 and result["status"] == "complete", "Worker failed; partial evidence retained")
    print("E1 " + args.phase + " complete in fresh process", flush=True)


if __name__ == "__main__":
    main()
