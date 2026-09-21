#!/usr/bin/env python3
"""Bounded, offline probe of the released native C2C fuser; no training.

All model inputs are synthetic fixtures. Assets must already exist locally.
Results measure output influence, not whole-agent or subjective continuity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


UPSTREAM_REVISION = "9fa85d35a20e20469af3e282926491222f4f11e0"
FUSER_DIRECTORY = "qwen3_0.6b+qwen2.5_0.5b_Fuser"
SEED = 20260921
KNOWN_NATIVE_TOKEN_EXCEPTIONS = {151667: "<think>", 151668: "</think>"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "assets", "fixtures", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def validate_source(upstream):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Upstream checkout does not match the required revision")
    sources = {}
    for name in (
        "rosetta/model/wrapper.py", "rosetta/model/projector.py",
        "rosetta/model/sampling.py", "rosetta/utils/registry.py",
    ):
        actual = (upstream / name).read_bytes()
        pinned = subprocess.check_output(
            ["git", "-C", str(upstream), "show", f"{revision}:{name}"]
        )
        if actual != pinned:
            raise ValueError(f"Required source has local changes: {name}")
        sources[name] = hashlib.sha256(actual).hexdigest()
    return sources


def prepare_fixtures(fixture, tokenizer, donor_tokenizer, torch):
    """Fail before scoring if a comparison would change token shape or options."""
    option_ids = {}
    for option in ("A", "B"):
        tokens = tokenizer.encode(option, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Option {option} is not exactly one receiver token")
        if donor_tokenizer.encode(option, add_special_tokens=False) != tokens:
            raise ValueError(f"Option {option} differs across the model tokenizers")
        option_ids[option] = tokens[0]
    if option_ids["A"] == option_ids["B"]:
        raise ValueError("Options collapse to one token")

    rendering = fixture["rendering"]

    def render(pair, value):
        content = rendering["template"].format(
            instruction=rendering["instruction"],
            record=pair["record"].format(value=value),
            question=pair["question"], choice_A=pair["choice_A"],
            choice_B=pair["choice_B"],
        )
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"]
        token_list = ids[0].tolist()
        exceptions, special_tokens = [], []
        for token_id in sorted(set(token_list)):
            receiver_token = tokenizer.convert_ids_to_tokens(token_id)
            donor_token = donor_tokenizer.convert_ids_to_tokens(token_id)
            receiver_special = token_id in tokenizer.all_special_ids
            donor_special = token_id in donor_tokenizer.all_special_ids
            if token_id in KNOWN_NATIVE_TOKEN_EXCEPTIONS:
                if (receiver_token != KNOWN_NATIVE_TOKEN_EXCEPTIONS[token_id]
                        or donor_token is not None or token_list.count(token_id) != 1):
                    raise ValueError("Known native thought-marker exception changed")
                exceptions.append({
                    "token_id": token_id, "receiver_token": receiver_token,
                    "donor_token": donor_token, "count": token_list.count(token_id),
                    "receiver_special": receiver_special, "donor_special": donor_special,
                })
                continue
            if (receiver_token != donor_token or receiver_special != donor_special
                    or tokenizer.decode([token_id], skip_special_tokens=False)
                    != donor_tokenizer.decode([token_id], skip_special_tokens=False)):
                raise ValueError(f"Unexpected receiver/donor token interpretation mismatch: {token_id}")
            if receiver_special:
                special_tokens.append({"token_id": token_id, "token": receiver_token,
                                       "count": token_list.count(token_id)})
        if {item["token_id"] for item in exceptions} != set(KNOWN_NATIVE_TOKEN_EXCEPTIONS):
            raise ValueError("Native nonthinking template no longer has exactly the expected markers")
        # Check continuation at the actual rendered boundary, not just in isolation.
        # The donor's textual prefix tokenization differs at the two known markers;
        # native C2C deliberately receives the receiver IDs at those positions.
        for label, current_tokenizer in (("receiver", tokenizer), ("donor", donor_tokenizer)):
            prefix_tokens = current_tokenizer.encode(formatted, add_special_tokens=False)
            for option, option_id in option_ids.items():
                continued = current_tokenizer.encode(formatted + option, add_special_tokens=False)
                if continued != prefix_tokens + [option_id]:
                    raise ValueError(f"Option {option} is not one appended token at the {label} boundary")
        return ids, {
            "known_native_token_exceptions": exceptions,
            "shared_special_tokens": special_tokens,
            "all_other_token_ids_match_token_strings_decoding_and_special_status": True,
            "ab_continuation_boundary_valid_for_both_tokenizers": True,
        }

    prepared = []
    validations = []
    seen = set()
    for pair in fixture["pairs"]:
        if pair["id"] in seen or len(pair["values"]) != 2:
            raise ValueError("Pairs need unique IDs and exactly two values")
        seen.add(pair["id"])
        if sorted(pair["answers"]) != ["A", "B"]:
            raise ValueError("Each twin pair must have opposite A/B answers")
        factual_rendered = [render(pair, value) for value in pair["values"]]
        redacted_rendered = [render(pair, rendering["missing_value"]) for _ in range(2)]
        factual = [item[0] for item in factual_rendered]
        redacted = [item[0] for item in redacted_rendered]
        shapes = {tuple(ids.shape) for ids in factual + redacted}
        if len(shapes) != 1 or factual[0].shape[1] < 2:
            raise ValueError(f"Unequal twin/redacted token shapes: {pair['id']}")
        if not torch.equal(redacted[0], redacted[1]):
            raise ValueError("Redacted receiver input depends on expected answer")
        changed = int((factual[0] != factual[1]).sum())
        if changed == 0:
            raise ValueError("Factual twins have identical input IDs")
        validations.append({
            "pair_id": pair["id"], "prompt_tokens": factual[0].shape[1],
            "changed_twin_tokens": changed, "redacted_twins_identical": True,
            "all_three_input_shapes_equal": True,
            "token_interpretation_checks": {
                "factual_twin_0": factual_rendered[0][1],
                "factual_twin_1": factual_rendered[1][1],
                "redacted": redacted_rendered[0][1],
            },
        })
        for index in range(2):
            prepared.append({
                "case_id": f"{pair['id']}:{index}", "pair_id": pair["id"],
                "family": pair["family"], "twin_index": index,
                "correct": pair["answers"][index],
                "twin_correct": pair["answers"][1 - index],
                "full_ids": factual[index], "twin_ids": factual[1 - index],
                "redacted_ids": redacted[index],
            })
    if len(prepared) != 24:
        raise ValueError("This fixed-protocol pilot requires exactly 24 cases")
    return prepared, option_ids, validations


def load_models(args, torch, transformers):
    from rosetta.model.projector import load_projector
    from rosetta.model.wrapper import RosettaModel

    dtype = getattr(torch, args.dtype)
    models = []
    for label in ("receiver", "donor"):
        print(f"Loading {label} locally on {args.device}/{args.dtype}", flush=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(args.assets / label), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=dtype,
            attn_implementation="eager",
        ).to(args.device).eval()
        model.requires_grad_(False)
        models.append(model)
    final = args.assets / "fuser" / FUSER_DIRECTORY / "final"
    indices = sorted(int(path.stem.split("_")[-1]) for path in final.glob("projector_[0-9]*.json"))
    if not indices or indices != list(range(len(indices))):
        raise ValueError("Projector files must be present and consecutively numbered")
    projectors, checkpoint_files = [], {}
    for index in indices:
        config_path = final / f"projector_{index}.json"
        weights_path = final / f"projector_{index}.pt"
        if json.loads(config_path.read_text())["class"] != "C2CProjector":
            raise ValueError("Checkpoint is not the native C2CProjector class")
        projector = load_projector(str(config_path), override_args={"dtype": dtype})
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        projector.load_state_dict(state, strict=True)
        projector.requires_grad_(False)
        projectors.append(projector)
        for path in (config_path, weights_path):
            with path.open("rb") as stream:
                checkpoint_files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    wrapper = RosettaModel(
        model_list=models, base_model_idx=0, projector_list=projectors,
        include_response=False, multi_source_fusion_mode="parallel",
    ).eval()
    wrapper.load_projector_config(str(final / "projector_config.json"))
    mapping = wrapper.projector_dict
    if set(mapping) != {0} or set(mapping[0]) != {1}:
        raise ValueError("Unexpected receiver/donor projector mapping")
    for target_layer, entries in mapping[0][1].items():
        if len(entries) != 1:
            raise ValueError("This probe requires one projector per mapped target layer")
        source_layer, projector_index = entries[0]
        if not (0 <= target_layer < models[0].config.num_hidden_layers
                and 0 <= source_layer < models[1].config.num_hidden_layers
                and 0 <= projector_index < len(projectors)):
            raise ValueError("Projector mapping refers to a nonexistent layer")
    gates = [{
        "projector_index": i,
        "key_gate_logit": float(p.key_gate_logit.detach().cpu()),
        "value_gate_logit": float(p.value_gate_logit.detach().cpu()),
        "key_gate_open": bool(p.key_gate_logit.detach().cpu() > 0),
        "value_gate_open": bool(p.value_gate_logit.detach().cpu() > 0),
    } for i, p in enumerate(wrapper.projector_list)]
    return wrapper, {
        "projector_count": len(projectors), "mapping": mapping, "gates": gates,
        "open_key_gates": sum(g["key_gate_open"] for g in gates),
        "open_value_gates": sum(g["value_gate_open"] for g in gates),
        "checkpoint_file_sha256": checkpoint_files,
        "models": [{
            "role": label, "model_type": model.config.model_type,
            "layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "kv_heads": model.config.num_key_value_heads,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        } for label, model in zip(("receiver", "donor"), models)],
    }


def score_logits(logits, option_ids, correct, tokenizer, torch):
    logits = logits.detach().float().cpu().reshape(-1)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    logprobs = torch.log_softmax(logits, dim=-1)
    ab_logits = logits[[option_ids["A"], option_ids["B"]]]
    forced = torch.softmax(ab_logits, dim=-1)
    absolute = logprobs[[option_ids["A"], option_ids["B"]]].exp()
    forced_choice = "AB"[int(forced.argmax())]
    greedy_id = int(logits.argmax())
    greedy_choice = next((k for k, v in option_ids.items() if v == greedy_id), None)
    return {
        "forced_ab_probability": dict(zip(("A", "B"), map(float, forced))),
        "unconditional_ab_probability": dict(zip(("A", "B"), map(float, absolute))),
        "unconditional_ab_mass": float(absolute.sum()),
        "forced_choice": forced_choice, "forced_correct": forced_choice == correct,
        "forced_probability_correct": float(forced["AB".index(correct)]),
        "unconditional_probability_correct": float(logprobs[option_ids[correct]].exp()),
        "unconditional_logprob_correct": float(logprobs[option_ids[correct]]),
        "greedy_next_token_id": greedy_id,
        "greedy_next_token_text": tokenizer.decode([greedy_id], skip_special_tokens=False),
        "greedy_choice": greedy_choice, "greedy_correct": greedy_choice == correct,
    }


class FusionProbe:
    """Observe native forward calls; optionally replace the donor output cache."""

    def __init__(self, wrapper, torch):
        self.wrapper, self.torch = wrapper, torch
        self.calls = []
        self.original_forwards = []
        for index, projector in enumerate(wrapper.projector_list):
            original = projector.forward
            self.original_forwards.append(original)

            def observed(source, target, *args, _index=index, _original=original, **kwargs):
                output = _original(source, target, *args, **kwargs)
                changes = []
                for before, after in zip(target, output):
                    delta = (after - before).detach().float()
                    if not torch.isfinite(delta).all():
                        raise ValueError("Nonfinite cache fusion")
                    norm = float(torch.linalg.vector_norm(delta))
                    baseline = float(torch.linalg.vector_norm(before.detach().float()))
                    changes.append({
                        "delta_l2": norm, "target_l2": baseline,
                        "relative_delta_l2": norm / baseline if baseline else None,
                        "max_abs_delta": float(delta.abs().max()),
                    })
                self.calls.append({"projector_index": _index, "key": changes[0], "value": changes[1]})
                return output

            projector.forward = observed

    def close(self):
        for projector, original in zip(self.wrapper.projector_list, self.original_forwards):
            projector.forward = original

    def run(self, receiver_ids, donor_ids, arm, replacement_cache=None):
        torch = self.torch
        self.calls = []
        donor_calls = []

        def donor_hook(module, inputs, output):
            from transformers.cache_utils import DynamicCache
            cache = output.past_key_values
            expected = receiver_ids.shape[1] - 1
            shapes = []
            for key, value in zip(cache.key_cache, cache.value_cache):
                if key.shape != value.shape or key.shape[2] != expected:
                    raise ValueError("Donor cache shape differs from the fixed prefix")
                if key.dtype != module.dtype or key.device != receiver_ids.device:
                    raise ValueError("Donor cache dtype/device mismatch")
                shapes.append(list(key.shape))
            donor_calls.append(shapes)
            if replacement_cache is not None:
                if len(replacement_cache.key_cache) != len(cache.key_cache):
                    raise ValueError("Injected cache has the wrong number of layers")
                copied = []
                for source_pair, actual_pair in zip(
                    zip(replacement_cache.key_cache, replacement_cache.value_cache),
                    zip(cache.key_cache, cache.value_cache),
                ):
                    for source, actual in zip(source_pair, actual_pair):
                        if (source.shape != actual.shape or source.dtype != actual.dtype
                                or source.device != actual.device):
                            raise ValueError("Injected cache shape/dtype/device differs from native")
                    copied.append(tuple(tensor.detach().clone() for tensor in source_pair))
                output.past_key_values = DynamicCache.from_legacy_cache(tuple(copied))
            elif arm == "zero_donor":
                output.past_key_values = DynamicCache.from_legacy_cache(tuple(
                    (torch.zeros_like(key), torch.zeros_like(value))
                    for key, value in zip(cache.key_cache, cache.value_cache)
                ))
            return output

        handle = self.wrapper.model_list[1].register_forward_hook(donor_hook)
        mask = -1 if arm == "no_fusion" else 1
        prefix = torch.tensor([mask, 0], dtype=torch.long, device=receiver_ids.device)
        prefix = prefix.repeat(receiver_ids.shape[1] - 1, 1).unsqueeze(0)
        final = torch.tensor([[[-1, 0]]], dtype=torch.long, device=receiver_ids.device)
        try:
            output = self.wrapper(
                input_ids=[receiver_ids, donor_ids],
                attention_mask=[torch.ones_like(receiver_ids), torch.ones_like(donor_ids)],
                kv_cache_index=[prefix, final], use_cache=True,
            )
            logits = output.logits[:, -1, :].detach().clone()
        finally:
            handle.remove()
        if len(donor_calls) != 1:
            raise ValueError("Expected exactly one native donor prefill call")
        expected_calls = 0 if arm == "no_fusion" else len(self.wrapper.projector_dict[0][1])
        if len(self.calls) != expected_calls:
            raise ValueError("Unexpected native projector call count")
        nonzero = sum(call["key"]["delta_l2"] > 0 or call["value"]["delta_l2"] > 0 for call in self.calls)
        return logits, {
            "projector_calls": len(self.calls), "nonzero_projector_calls": nonzero,
            "donor_cache_shapes": donor_calls[0], "per_projector": self.calls,
        }


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable automatic MPS fallback; backend changes must be explicit")
    import torch
    import transformers

    if torch.__version__.split("+")[0] != "2.6.0" or transformers.__version__ != "4.52.4":
        raise ValueError("Use upstream-pinned torch 2.6.0 and transformers 4.52.4")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS backend is unavailable; rerun explicitly with --device cpu")
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    source_digests = validate_source(args.upstream)
    sys.path.insert(0, str(args.upstream))
    fixture_bytes = args.fixtures.read_bytes()
    fixture = json.loads(fixture_bytes)
    tokenizer, donor_tokenizer = [transformers.AutoTokenizer.from_pretrained(
        str(args.assets / label), local_files_only=True, trust_remote_code=False
    ) for label in ("receiver", "donor")]
    cases, option_ids, validations = prepare_fixtures(fixture, tokenizer, donor_tokenizer, torch)
    print(f"Validated all {len(cases)} cases before scoring", flush=True)
    wrapper, model_metadata = load_models(args, torch, transformers)
    for case in cases:
        for label in ("full_ids", "twin_ids", "redacted_ids"):
            if int(case[label].max()) >= min(model.config.vocab_size for model in wrapper.model_list):
                raise ValueError("Receiver token IDs exceed a model vocabulary")
            case[label] = case[label].to(args.device)
    report = {
        "schema_version": 1, "status": "RUNNING", "seed": SEED,
        "scope": "Synthetic output fidelity and factual influence; not whole-agent restoration",
        "upstream_revision": UPSTREAM_REVISION, "source_sha256": source_digests,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "runtime": {
            "python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "device": args.device, "dtype": args.dtype,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "attention_implementation": "eager", "automatic_backend_fallback": False,
            "eval_mode": True, "sampling": "none; next-token logits only",
            "deterministic_algorithms_enforced": torch.are_deterministic_algorithms_enabled(),
            "numerical_scope": "Fixed backend and dtype; no cross-device bitwise equivalence claim",
            "latency_scope": "Includes native prefill, cache copies and instrumentation; not a throughput benchmark",
        },
        "scoring": {
            "option_ids": option_ids, "forced_choice": "argmax over bare A and B token logits",
            "absolute_probabilities": "softmax over full vocabulary; forced probabilities condition on A/B",
            "receiver_template_for_both_models": True, "enable_thinking": False,
            "native_tokenizer_limitations": {
                "151667": "Receiver <think> token has no donor tokenizer entry",
                "151668": "Receiver </think> token has no donor tokenizer entry",
                "interpretation": "Released training/inference passes receiver IDs to both models. Preserve these two explicit exceptions; require all other used token meanings and special-token status to agree.",
                "marker_count_per_prompt": {"151667": 1, "151668": 1},
            },
            "prefix_partition": "N-1 fused positions followed by one receiver-only position",
            "private_fact_probe": "Out-of-distribution intervention: receiver fact redacted; donor factual",
            "zero_donor": "Zero cache still enters learned fuser; not a no-fusion baseline",
        },
        "validation": validations, "checkpoint": model_metadata,
        "controls": {}, "references": [], "rows": [],
    }
    atomic_write(args.output, report)
    probe = FusionProbe(wrapper, torch)

    def synchronize():
        if args.device == "mps":
            torch.mps.synchronize()

    try:
        with torch.inference_mode():
            # Establish that injecting an independently computed matched cache
            # reproduces the native fusion path before any scored ablations.
            first_ids = cases[0]["full_ids"]
            independent = wrapper.model_list[1](
                input_ids=first_ids[:, :-1], attention_mask=torch.ones_like(first_ids[:, :-1]),
                position_ids=None, use_cache=True, return_dict=True,
            ).past_key_values
            native_logits, native_fusion = probe.run(first_ids, first_ids, "matched_donor")
            injected_logits, injected_fusion = probe.run(
                first_ids, first_ids, "matched_donor", replacement_cache=independent,
            )
            native_cpu, injected_cpu = native_logits.cpu(), injected_logits.cpu()
            difference = (native_cpu - injected_cpu).double()
            native_probability = native_cpu.double().softmax(dim=-1)
            injected_probability = injected_cpu.double().softmax(dim=-1)
            exactly_equal = torch.equal(native_cpu, injected_cpu)
            report["controls"]["independent_matched_cache_injection"] = {
                "case_id": cases[0]["case_id"], "before_scored_ablations": True,
                "max_absolute_logit_difference": float(difference.abs().max()),
                "total_variation_distance": float((native_probability - injected_probability).abs().sum() / 2),
                "greedy_next_token_equal": bool(native_cpu.argmax() == injected_cpu.argmax()),
                "logits_exactly_equal": exactly_equal,
                "acceptance_rule": "Exact full-logit equality on this fixed backend/dtype",
                "native_projector_calls": native_fusion["projector_calls"],
                "injected_projector_calls": injected_fusion["projector_calls"],
            }
            atomic_write(args.output, report)
            if not exactly_equal:
                raise ValueError("Independent matched-cache injection failed exact reproducibility")
            del independent, native_logits, injected_logits, native_cpu, injected_cpu
            del native_probability, injected_probability, difference
            print("Independent matched-cache injection exactly reproduces native logits", flush=True)
            for case_index, case in enumerate(cases):
                common = {k: case[k] for k in ("case_id", "pair_id", "family", "twin_index", "correct", "twin_correct")}
                reference_logits = {}
                for model_index, label in enumerate(("receiver", "donor")):
                    synchronize()
                    start = time.perf_counter()
                    output = wrapper.model_list[model_index](
                        input_ids=case["full_ids"], attention_mask=torch.ones_like(case["full_ids"]),
                        use_cache=True, return_dict=True,
                    )
                    logits = output.logits[:, -1, :].detach().clone()
                    reference_logits[label] = logits
                    del output
                    synchronize()
                    report["references"].append({
                        **common, "arm": f"{label}_full_context_reference",
                        "latency_seconds": time.perf_counter() - start,
                        **score_logits(logits, option_ids, case["correct"], tokenizer if label == "receiver" else donor_tokenizer, torch),
                    })
                for condition in ("shared_context", "private_fact_probe"):
                    receiver_ids = case["full_ids"] if condition == "shared_context" else case["redacted_ids"]
                    arm_shapes = None
                    for arm in ("no_fusion", "matched_donor", "twin_swapped_donor", "zero_donor"):
                        donor_ids = case["twin_ids"] if arm == "twin_swapped_donor" else case["full_ids"]
                        synchronize()
                        start = time.perf_counter()
                        logits, fusion = probe.run(receiver_ids, donor_ids, arm)
                        synchronize()
                        elapsed = time.perf_counter() - start
                        if arm_shapes is not None and fusion["donor_cache_shapes"] != arm_shapes:
                            raise ValueError("Donor cache shapes changed across ablation arms")
                        arm_shapes = fusion["donor_cache_shapes"]
                        report["rows"].append({
                            **common, "condition": condition, "arm": arm,
                            "prompt_tokens": receiver_ids.shape[1], "latency_seconds": elapsed,
                            **score_logits(logits, option_ids, case["correct"], tokenizer, torch),
                            "fusion": fusion,
                        })
                        if case_index == 0 and condition == "shared_context" and arm == "no_fusion":
                            difference = (logits - reference_logits["receiver"]).float()
                            report["controls"]["native_chunked_no_fusion_vs_full_prefill"] = {
                                "case_id": case["case_id"],
                                "max_absolute_logit_difference": float(difference.abs().max()),
                                "logit_l2_difference": float(torch.linalg.vector_norm(difference)),
                                "greedy_next_token_equal": bool(logits.argmax() == reference_logits["receiver"].argmax()),
                                "interpretation": "Measures chunking/backend rounding, not C2C influence",
                            }
                    atomic_write(args.output, report)
                    print(f"{case_index + 1}/{len(cases)} {case['case_id']} {condition}: four arms saved", flush=True)
                del reference_logits
        report["status"] = "COMPLETE"
        report["counts"] = {"cases": len(cases), "references": len(report["references"]), "rows": len(report["rows"])}
        atomic_write(args.output, report)
        print(json.dumps({"status": report["status"], **report["counts"]}), flush=True)
    except BaseException as error:
        report["status"] = "FAILED_PARTIAL"
        report["failure"] = {"exception_type": type(error).__name__}
        atomic_write(args.output, report)
        raise
    finally:
        probe.close()


if __name__ == "__main__":
    main()
