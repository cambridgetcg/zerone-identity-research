"""Native receiver KV serialization control; synthetic inference state only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def verify_receiver(model_path):
    manifest_path = Path(__file__).resolve().parents[2]/"evidence/c2c-fidelity-sources.json"
    assets = json.loads(manifest_path.read_text())["assets"]
    receiver = next(asset for asset in assets if asset["alias"] == "receiver")
    if (receiver["repo"] != "Qwen/Qwen3-0.6B"
            or receiver["revision"] != "c1899de289a04d12100db370d81485cdf75e47ca"):
        raise ValueError("Manifest does not identify the fixed receiver checkpoint")
    for item in receiver["files"]:
        path = model_path/item["path"]
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != item["bytes"] or observed != item["sha256"]:
            raise ValueError("Receiver asset differs from pinned bytes: " + item["path"])


def copy_cache(cache):
    return DynamicCache.from_legacy_cache(tuple(
        (key.detach().clone(), value.detach().clone())
        for key, value in cache.to_legacy_cache()
    ))


def compare_logits(first, second):
    a, b = first.detach().cpu().double(), second.detach().cpu().double()
    pa, pb = a.softmax(-1), b.softmax(-1)
    return {
        "maximum_absolute_logit_difference": (a-b).abs().max().item(),
        "total_variation_distance": (pa-pb).abs().sum().item()/2,
        "greedy_token_equal": bool(a.argmax()==b.argmax()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_receiver(args.model)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable automatic MPS fallback; backend changes must be explicit")
    if torch.__version__.split("+")[0] != "2.6.0" or transformers.__version__ != "4.52.4":
        raise ValueError("Use the pinned inference package versions")
    torch.manual_seed(20260921)
    torch.set_num_threads(6)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, attn_implementation="eager"
    ).to(args.device).eval()
    prompt = "The sealed envelope contains the code 42. Copy only the code."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer(rendered, return_tensors="pt")["input_ids"].to(args.device)
    length = ids.shape[1]
    mask = torch.ones_like(ids)
    positions = torch.arange(length, device=args.device)[None, :]

    with torch.inference_mode():
        full = model(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        full_logits = full.logits[:, -1, :].detach().clone()
        del full
        prefix = model(ids[:, :-1], attention_mask=mask[:, :-1],
                       position_ids=positions[:, :-1], use_cache=True)
        before = prefix.past_key_values
        saved = {f"{i}.{kind}": tensor.detach().cpu().clone()
                 for i, pair in enumerate(before.to_legacy_cache())
                 for kind, tensor in zip(("key", "value"), pair)}
        with tempfile.TemporaryDirectory(prefix="c2c-checkpoint-") as folder:
            checkpoint = Path(folder)/"synthetic-kv.pt"
            torch.save(saved, checkpoint)
            checkpoint_bytes = checkpoint.stat().st_size
            checkpoint_hash = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        equal = all(torch.equal(saved[key], loaded[key]) for key in saved)
        restored = DynamicCache.from_legacy_cache(tuple(
            (loaded[f"{i}.key"].to(args.device), loaded[f"{i}.value"].to(args.device))
            for i in range(len(before))
        ))
        assert restored.get_seq_length()==before.get_seq_length()==length-1
        resumed = model(ids[:, -1:], attention_mask=mask,
                        position_ids=positions[:, -1:], past_key_values=restored,
                        use_cache=True)
        uninterrupted = model(ids[:, -1:], attention_mask=mask,
                              position_ids=positions[:, -1:],
                              past_key_values=copy_cache(before), use_cache=True)
        serialization_comparison = compare_logits(uninterrupted.logits[:, -1, :],
                                                 resumed.logits[:, -1, :])
        full_comparison = compare_logits(full_logits, resumed.logits[:, -1, :])
        del before, prefix, saved, loaded
        branch_outputs = [uninterrupted, resumed]
        trajectories = [[], []]
        step_differences = []
        for step in range(4):
            step_differences.append(compare_logits(
                branch_outputs[0].logits[:, -1, :], branch_outputs[1].logits[:, -1, :]
            ))
            next_ids = [out.logits[:, -1, :].argmax(-1).reshape(1, 1)
                        for out in branch_outputs]
            for index, token in enumerate(next_ids):
                trajectories[index].append(int(token.item()))
            if step==3:
                break
            branch_outputs = [model(
                next_ids[index], attention_mask=torch.ones((1, length+step+1),
                                                          device=args.device, dtype=torch.long),
                position_ids=torch.tensor([[length+step]], device=args.device),
                past_key_values=out.past_key_values, use_cache=True,
            ) for index, out in enumerate(branch_outputs)]

    result = {
        "kind": "native_receiver_checkpoint_control",
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "device": args.device,
        "dtype": "float32", "attention": "eager", "sampling": "greedy",
        "prompt": prompt, "rendered_prompt": rendered, "input_token_ids": ids.cpu().tolist()[0],
        "serialized_tensor_count": 2*model.config.num_hidden_layers,
        "serialized_bytes": checkpoint_bytes, "serialized_sha256": checkpoint_hash,
        "all_reloaded_tensors_bitwise_equal": equal,
        "same_chunking_uninterrupted_vs_restored": serialization_comparison,
        "full_prefill_vs_restored": full_comparison,
        "greedy_continuation_ids": trajectories,
        "greedy_continuation_text": [tokenizer.decode(tokens) for tokens in trajectories],
        "continuation_comparisons": step_differences,
        "elapsed_seconds": time.monotonic()-started,
        "interpretation": "Only native inference-state serialization under this fixed setup; no cross-model or subjective continuity claim.",
        "tolerance_policy": "Require exact tensor equality and exact same-chunking logits. Full-prefill differences are reported without assuming bitwise equality across chunking. Four fixed greedy steps include any end tokens; no sampler RNG claim.",
    }
    result["passed"] = bool(equal and serialization_comparison["maximum_absolute_logit_difference"]==0
                            and trajectories[0]==trajectories[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("Native checkpoint control did not meet its exact-equality criterion")


if __name__=="__main__":
    main()
