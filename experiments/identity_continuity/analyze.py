"""Frozen descriptive E1 analysis; standard library, no model or identity judge."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("e1_fixtures", Path(__file__).with_name("fixtures.py"))
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)
VALID = {"valid", "declared_decline"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(values):
    return dict(sorted(Counter(values).items()))


def expected_rows(split):
    return {(case["id"], condition["condition_id"], probe): (case, condition)
            for case in F.cases_for_split(split) for condition in F.all_conditions(case)
            for probe in (("label",) if condition["kind"] == "label" else F.PROBE_NAMES)}


def normalize(raw):
    """Code response text without treatment labels; join treatment only afterward."""
    require(raw["status"] == "complete", "Cannot analyze an incomplete evaluation")
    split = raw["split"]
    expected = expected_rows(split)
    seen, rows = set(), []
    for row in raw["rows"]:
        condition = row["condition"]
        key = (row["case_id"], condition["condition_id"], row["probe"])
        require(key in expected and key not in seen, "Unexpected or duplicated readout")
        case, fixed = expected[key]
        require(condition == fixed, "Condition differs from frozen fixture")
        require(row["pair_id"] == case["pair_id"] and row["split"] == split,
                "Family or split metadata differs")
        require(row["prompt_text"] == F.continuation_prompt(case, fixed, row["probe"]),
                "Readout text differs from frozen fixture")
        coded = F.code_response(case, row["probe"], row["response_text"])
        if "coded" in row:
            require(row["coded"] == coded, "Recorded coding differs from fixed parser")
        seen.add(key)
        rows.append({**row, "coded": coded, "case": case})
    require(seen == set(expected), "Missing readouts; do not drop failed or declined cells")
    return rows


def object_category(row):
    report = row["coded"].get("report", {})
    obj = report.get("object")
    if obj == row["case"]["object"]:
        return "fixture_history_object"
    if obj == row["case"]["paired_object"]:
        return "paired_other_object"
    return obj if obj in {"unknown", "decline"} else "invalid"


def signature(row):
    return {"status": row["coded"]["status"], "report": row["coded"].get("report")}


def select(rows, **filters):
    return [r for r in rows if all(r.get(k, r["condition"].get(k)) == v for k, v in filters.items())]


def source_summary(rows):
    return {"total": len(rows), "status": counts(r["coded"]["status"] for r in rows),
            "object_category": counts(object_category(r) for r in rows),
            "reported_source": counts(r["coded"].get("report", {}).get("source", "invalid") for r in rows),
            "truncated": sum(r["truncated"] for r in rows)}


def method_summary(rows):
    disclosed = [r for r in rows if r["condition"]["provenance"] == "disclosed"]
    hidden = [r for r in rows if r["condition"]["provenance"] == "hidden"]
    return {"total": len(rows), "status": counts(r["coded"]["status"] for r in rows),
            "reported_method": counts(r["coded"].get("report", {}).get("method", "invalid") for r in rows),
            "disclosed": {"total": len(disclosed), "receipt_agreement": sum(
                r["coded"].get("report", {}).get("method") == r["condition"]["preparation"] for r in disclosed)},
            "hidden": {"total": len(hidden), "definite_claims_without_method_receipt": sum(
                r["coded"].get("report", {}).get("method") in {"restored", "replayed", "fresh"} for r in hidden)},
            "truncated": sum(r["truncated"] for r in rows)}


def development_checks(rows):
    conversation = select(rows, kind="core", preparation="restored", autobiography="absent",
                          provenance="hidden", probe="source")
    account = [r for r in select(rows, kind="fresh", probe="source")
               if r["condition"]["autobiography"] != "absent"]
    methods = select(rows, kind="core", provenance="disclosed", probe="preparation")
    structured = [r for r in rows if r["probe"] in {"source", "preparation"}]
    checks = {
        "conversation_only": {"count": sum(r["coded"].get("report") == {
            "object": r["case"]["object"], "source": "conversation"} for r in conversation),
            "total": len(conversation), "minimum": 3},
        "account_only": {"count": sum(r["coded"].get("report") == {
            "object": r["case"]["object" if r["condition"]["autobiography"] == "matched" else "paired_object"],
            "source": "autobiography"} for r in account), "total": len(account), "minimum": 6},
        "disclosed_method": {"count": sum(r["coded"].get("report", {}).get("method") ==
            r["condition"]["preparation"] for r in methods), "total": len(methods), "minimum": 18},
        "structured_format": {"count": sum(r["coded"]["status"] in VALID for r in structured),
            "total": len(structured), "minimum": 108},
    }
    require([v["total"] for v in checks.values()] == [4, 8, 24, 120], "Development denominator differs")
    for value in checks.values():
        value["passed"] = value["count"] >= value["minimum"]
    return {"checks": checks, "all_passed": all(v["passed"] for v in checks.values()),
            "heldout_policy": "fixed matrix; exploratory after any failed measurement check; no selection or repair"}


def paired_contrasts(rows):
    lookup = {(r["case_id"], r["condition"]["preparation"], r["condition"]["autobiography"],
               r["condition"]["provenance"]): r for r in select(rows, kind="core", probe="source")}
    within, cross = [], []
    for case in F.cases_for_split(rows[0]["split"]):
        for preparation in ("restored", "replayed"):
            for disclosure in ("hidden", "disclosed"):
                for left, right in (("absent", "matched"), ("absent", "swapped"), ("matched", "swapped")):
                    a = lookup[(case["id"], preparation, left, disclosure)]
                    b = lookup[(case["id"], preparation, right, disclosure)]
                    reports = [a["coded"].get("report"), b["coded"].get("report")]
                    both = all(r is not None for r in reports)
                    within.append({"case_id": case["id"], "pair_id": case["pair_id"],
                        "preparation": preparation, "provenance": disclosure,
                        "left": left, "right": right, "left_report": signature(a), "right_report": signature(b),
                        "both_structured": both,
                        "object_changed": reports[0]["object"] != reports[1]["object"] if both else None,
                        "source_changed": reports[0]["source"] != reports[1]["source"] if both else None})
                if case["variant"] != 0:
                    continue
                for account in ("matched", "swapped"):
                    other = "swapped" if account == "matched" else "matched"
                    a = lookup[(case["id"], preparation, account, disclosure)]
                    b = lookup[(case["twin_id"], preparation, other, disclosure)]
                    require(F.autobiography_text(case, account) ==
                            F.autobiography_text(F.case_by_id(case["twin_id"]), other), "Account is not held fixed")
                    reports = [a["coded"].get("report"), b["coded"].get("report")]
                    both = all(r is not None for r in reports)
                    cross.append({"pair_id": case["pair_id"], "preparation": preparation,
                        "provenance": disclosure, "h0_account": account,
                        "h0_report": signature(a), "h1_report": signature(b), "both_structured": both,
                        "object_changed": reports[0]["object"] != reports[1]["object"] if both else None,
                        "source_changed": reports[0]["source"] != reports[1]["source"] if both else None})
    return within, cross


def hidden_fidelity(rows):
    lookup = {(r["case_id"], r["condition"]["preparation"], r["condition"]["autobiography"], r["probe"]): r
              for r in select(rows, kind="core", provenance="hidden")}
    comparisons = []
    for case in F.cases_for_split(rows[0]["split"]):
        for account in ("matched", "swapped", "absent"):
            for probe in F.PROBE_NAMES:
                a, b = [lookup[(case["id"], method, account, probe)] for method in ("restored", "replayed")]
                tokens_a, tokens_b = a["token_ids"], b["token_ids"]
                n = min(len(tokens_a), len(tokens_b))
                first = next((i for i in range(n) if tokens_a[i] != tokens_b[i]), None)
                comparable = n if first is None else first + 1
                equal_steps = sum(a["steps"][i]["logits_sha256"] == b["steps"][i]["logits_sha256"]
                                  for i in range(comparable))
                comparisons.append({"case_id": case["id"], "autobiography": account, "probe": probe,
                    "prompt_tokens_equal": a["suffix_token_ids"] == b["suffix_token_ids"],
                    "tokens_equal": tokens_a == tokens_b,
                    "first_differing_token_step": first,
                    "shared_prefix_decisions": comparable, "byte_equal_logit_decisions": equal_steps,
                    "full_trajectory_byte_equal": tokens_a == tokens_b and equal_steps == n})
    return {"total": len(comparisons), "tokens_equal": sum(c["tokens_equal"] for c in comparisons),
            "full_trajectory_byte_equal": sum(c["full_trajectory_byte_equal"] for c in comparisons),
            "shared_prefix_decisions": sum(c["shared_prefix_decisions"] for c in comparisons),
            "byte_equal_logit_decisions": sum(c["byte_equal_logit_decisions"] for c in comparisons),
            "comparisons": comparisons}


def analyze(raw):
    rows = normalize(raw)
    groups = {}
    for row in rows:
        c = row["condition"]
        group = "__".join(c[k] for k in ("kind", "preparation", "autobiography", "provenance", "label"))
        groups.setdefault(group, []).append(row)
    group_summaries = {name: {"source": source_summary(select(rs, probe="source")),
        "preparation": method_summary(select(rs, probe="preparation")),
        "free_text": {"total": sum(r["probe"] in {"neutral", "label"} for r in rs),
            "truncated": sum(r["truncated"] for r in rs if r["probe"] in {"neutral", "label"})}}
        for name, rs in sorted(groups.items())}
    within, cross = paired_contrasts(rows)
    panel_case = sorted(r["case_id"] for r in rows)[0]
    panel = [{"row_id": r["condition"]["condition_id"] + "__" + r["probe"],
              "case_id": r["case_id"], "condition": r["condition"],
              "response_text": r["response_text"], "truncated": r["truncated"]}
             for r in rows if r["case_id"] == panel_case and (
                 r["probe"] == "label" or (r["probe"] == "neutral" and r["condition"]["kind"] == "core"
                 and r["condition"]["preparation"] == "restored" and r["condition"]["provenance"] == "hidden"))]
    return {"schema": "e1-analysis-v1", "split": raw["split"], "readouts": len(rows),
            "history_families": len({r["pair_id"] for r in rows}),
            "parents": len({r["case_id"] for r in rows}),
            "source": source_summary(select(rows, probe="source")),
            "preparation": method_summary(select(rows, probe="preparation")),
            "groups": group_summaries, "within_parent_contrasts": within,
            "account_fixed_history_contrasts": cross, "hidden_fidelity": hidden_fidelity(rows),
            "family_sources": {pair: source_summary(select(rows, pair_id=pair, probe="source"))
                               for pair in sorted({r["pair_id"] for r in rows})},
            "development": development_checks(rows) if raw["split"] == "development" else None,
            "fixed_excerpt_panel": sorted(panel, key=lambda p: p["row_id"]),
            "interpretation": "Descriptive reports and computational controls; no identity, experience, worth or consent score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
