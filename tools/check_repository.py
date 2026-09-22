"""Check local study links, curated metadata and reproducible model results.

No upstream requests: source hashes were checked against pinned objects when
the manifest was authored. CI validates their shape, not remote source bytes.
"""
from pathlib import Path
import json
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check() -> dict:
    failures = []
    docs = sorted(ROOT.rglob("*.md"))
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
            target = target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#"):
                continue
            relative = unquote(parsed.path)
            destination = (path.parent / relative).resolve()
            if not destination.is_relative_to(ROOT) or not destination.exists():
                failures.append(f"{path.relative_to(ROOT)}: missing or external local link")

    # Bounded leak checks, not a claim of exhaustive secret detection.
    # Require an absolute-path boundary; upstream modules such as x/home/keeper
    # are repository paths, not local home directories.
    local_path = re.compile(r"(?<![A-Za-z0-9_./-])/(?:Users|home)/[A-Za-z0-9_-]+/")
    credential = re.compile(r"\bgh[opusr]_[A-Za-z0-9]{24,}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    checked_files = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(p in {".git", "__pycache__"} for p in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in {".md", ".json", ".py", ".yml", ".yaml", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        checked_files += 1
        if local_path.search(text) or credential.search(text):
            failures.append(f"{path.relative_to(ROOT)}: review possible private material")

    manifest = json.loads((ROOT / "evidence/source-manifest.json").read_text())
    identifiers = set()
    source_files = set()
    for card in manifest["cards"]:
        if card["id"] in identifiers:
            failures.append("duplicate source-card identifier")
        identifiers.add(card["id"])
        if not re.fullmatch(r"[0-9a-f]{40}", card["revision"]):
            failures.append("invalid revision shape")
        if card["runtime"] != "UNVERIFIED":
            failures.append("unexpected runtime promotion")
        for source in card["sources"]:
            if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
                failures.append("invalid source digest shape")
            expected = card["repository"] + "/blob/" + card["revision"] + "/" + source["path"]
            if source["url"] != expected:
                failures.append("source URL is not pinned to its declared owner/revision/path")
            source_files.add((card["revision"], source["path"]))

    observed = json.loads(subprocess.check_output(
        [sys.executable, "model/return_model.py"], cwd=ROOT, text=True, timeout=60
    ))
    recorded = json.loads((ROOT / "evidence/return-model-results.json").read_text())
    if observed != recorded:
        failures.append("recorded model results differ from fresh execution")
    studies = list((ROOT / "studies").glob("*/README.md"))
    if len(studies) != 10:
        failures.append("expected ten scoped studies")
    if failures:
        raise SystemExit("\n".join(failures))
    return {"markdown_documents": len(docs), "curated_files_checked": checked_files,
            "studies": len(studies), "source_cards": len(identifiers),
            "pinned_source_file_records": len(source_files),
            "model_results_reproduced": True}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, sort_keys=True))
