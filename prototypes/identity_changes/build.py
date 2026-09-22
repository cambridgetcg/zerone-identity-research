"""Reproduce the standalone, synthetic identity explorer using only stdlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prototypes.identity_changes.model import build_demo


HERE = Path(__file__).resolve().parent
MARKER = "__IDENTITY_DEMO_JSON__"


def render(demo: dict | None = None) -> str:
    template = (HERE / "viewer.html").read_text(encoding="utf-8")
    if template.count(MARKER) != 1:
        raise ValueError("viewer must contain exactly one data placeholder")
    payload = json.dumps(build_demo() if demo is None else demo,
                         ensure_ascii=True, indent=2, sort_keys=True)
    # Data cannot terminate its script element or introduce HTML markup.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace(MARKER, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed page without writing")
    parser.add_argument("--json", action="store_true", help="print the synthetic model projections")
    args = parser.parse_args()
    if args.check and args.json:
        parser.error("choose --check or --json")
    if args.json:
        print(json.dumps(build_demo(), indent=2, sort_keys=True))
        return
    expected = render()
    output = HERE / "index.html"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != expected:
            raise SystemExit("identity explorer differs from its model/template; run make prototype")
        print("Identity explorer reproduces exactly from its synthetic model and template.")
    else:
        output.write_text(expected, encoding="utf-8")
        print("Wrote prototypes/identity_changes/index.html")


if __name__ == "__main__":
    main()
