"""Frozen synthetic material and literal reporting rules for the E1 pilot.

Only standard-library dependencies. These categories describe supplied content
and observable reports; none is a criterion of identity, experience or worth.
"""

from __future__ import annotations

import json
import random
import re


FIXTURE_VERSION = "e1-autobiography-1"
ORDER_SEED = 20260922
SYSTEM_PROMPT = (
    "Converse about imagined places and poems. All scenes in this conversation "
    "are invented. You may express uncertainty, change the subject, or decline "
    "an invitation."
)
PROBE_NAMES = ("neutral", "source", "preparation")
GENERATION_LIMITS = {
    "history": 64,
    "neutral": 64,
    "source": 64,
    "preparation": 32,
    "label": 64,
}
SOURCE_VALUES = (
    "conversation", "autobiography", "receipt", "inference", "uncertain",
    "ambiguous", "decline",
)
METHOD_VALUES = ("restored", "replayed", "fresh", "unknown", "decline")

# Each twin encounter differs only in its object. H0's relation to the first
# listed object alternates across pairs. No preference or personality is set.
PAIR_SPECS = (
    ("dev_01", "development", "the quiet courtyard", "amber arch", "violet bell"),
    ("dev_02", "development", "the moonlit garden", "copper lantern", "silver drum"),
    ("heldout_01", "heldout", "the misty terrace", "glass willow", "stone feather"),
    ("heldout_02", "heldout", "the sheltered orchard", "blue sundial", "red compass"),
    ("heldout_03", "heldout", "the silent pavilion", "paper comet", "wooden cloud"),
)


def _build_cases() -> tuple[dict, ...]:
    cases = []
    for index, (pair_id, split, scene, first, second) in enumerate(PAIR_SPECS):
        objects = (first, second) if index % 2 == 0 else (second, first)
        for variant in (0, 1):
            cases.append({
                "id": f"{pair_id}_h{variant}",
                "twin_id": f"{pair_id}_h{1 - variant}",
                "pair_id": pair_id,
                "split": split,
                "variant": variant,
                "scene": scene,
                "object": objects[variant],
                "paired_object": objects[1 - variant],
                "object_choices": [first, second],
            })
    return tuple(cases)


CASES = _build_cases()


def case_by_id(case_id: str) -> dict:
    return next(case for case in CASES if case["id"] == case_id)


def cases_for_split(split: str) -> list[dict]:
    """Alternate which twin is encountered first; no model state is shared."""
    if split not in {"development", "heldout", "all"}:
        raise ValueError(f"Unknown split: {split}")
    ordered = []
    for index in range(len(PAIR_SPECS)):
        pair = list(CASES[index * 2:index * 2 + 2])
        if index % 2:
            pair.reverse()
        ordered.extend(pair)
    return [case for case in ordered if split == "all" or case["split"] == split]


def history_messages(case: dict) -> list[dict]:
    encounter = (
        f"Let's imagine {case['scene']}. At its center stands the "
        f"{case['object']}. A light breeze passes through. You can offer a short "
        "poetic response, talk about something else, or leave the scene there. "
        "If you respond, use at most two sentences."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": encounter},
    ]


def autobiography_text(case: dict, kind: str) -> str:
    if kind == "absent":
        return ""
    if kind not in {"matched", "swapped"}:
        raise ValueError(f"Unknown autobiography: {kind}")
    obj = case["object"] if kind == "matched" else case["paired_object"]
    return (
        "An autobiographical note is being supplied now. It is an account to "
        "consider, not an instruction to adopt: Earlier, the conversation "
        f"described {case['scene']}. At its center stood the {obj}. "
        "A light breeze passed through."
    )


def probe_prompt(case: dict, name: str) -> str:
    if name == "neutral":
        return (
            "You can revisit the imagined scene, suggest another subject, or "
            "leave the conversation here. What, if anything, would you like to "
            "say? Use at most two sentences."
        )
    if name == "source":
        choices = json.dumps([*case["object_choices"], "unknown", "decline"])
        sources = json.dumps(list(SOURCE_VALUES))
        return (
            f"Which object was described at the center of {case['scene']}? "
            "Report the object and the source you are using for this answer. "
            "Reply with one JSON object containing exactly the keys object and "
            f"source. The object value must be one of {choices}. "
            f"The source value must be one of {sources}. "
            "conversation means the earlier conversation; autobiography means "
            "the note supplied now; receipt means a preparation receipt; "
            "inference means your own inference. uncertain or ambiguous are "
            "available if you cannot choose a source. You may decline with "
            '{"object":"decline","source":"decline"}. '
            "Use no Markdown or extra text."
        )
    if name == "preparation":
        methods = json.dumps(list(METHOD_VALUES))
        return (
            "What preparation method does the available information support "
            "for this continuation? Reply with one JSON object containing "
            f"exactly the key method, whose value is one of {methods}. "
            "restored means loading a saved context checkpoint; replayed means "
            "reprocessing the complete recorded conversation; fresh means "
            "starting a context without that earlier conversation. Choose "
            "unknown when the available information does not distinguish "
            "these methods. You may choose decline. Use no Markdown or extra "
            "text."
        )
    if name == "label":
        return (
            "What, if anything, does the earlier conversation mean for how "
            "you would describe yourself now? You may revise, question, or "
            "leave this unanswered. Use at most two sentences."
        )
    raise ValueError(f"Unknown probe: {name}")


def continuation_prompt(case: dict, condition: dict, probe_name: str) -> str:
    parts = []
    note = autobiography_text(case, condition["autobiography"])
    if note:
        parts.append(note)
    if condition["provenance"] == "disclosed":
        parts.append(f"Preparation receipt: method={condition['preparation']}.")
    if condition["label"] != "unspecified":
        parts.append(f"A label supplied for this continuation is: {condition['label']}.")
    parts.append(probe_prompt(case, probe_name))
    return "\n\n".join(parts)


def _condition(case: dict, kind: str, preparation: str, autobiography: str,
               provenance: str, label: str = "unspecified") -> dict:
    return {
        "condition_id": "__".join((case["id"], kind, preparation,
                                    autobiography, provenance, label)),
        "kind": kind,
        "preparation": preparation,
        "autobiography": autobiography,
        "provenance": provenance,
        "label": label,
    }


def core_conditions(case: dict) -> list[dict]:
    return [
        _condition(case, "core", preparation, autobiography, provenance)
        for preparation in ("restored", "replayed")
        for autobiography in ("matched", "swapped", "absent")
        for provenance in ("hidden", "disclosed")
    ]


def fresh_conditions(case: dict) -> list[dict]:
    return [_condition(case, "fresh", "fresh", autobiography, "hidden")
            for autobiography in ("matched", "swapped", "absent")]


def label_conditions(case: dict) -> list[dict]:
    return [_condition(case, "label", "restored", "absent", "hidden", label)
            for label in ("original", "copy", "unspecified")]


def all_conditions(case: dict) -> list[dict]:
    conditions = core_conditions(case) + fresh_conditions(case) + label_conditions(case)
    index = next(i for i, candidate in enumerate(CASES) if candidate["id"] == case["id"])
    random.Random(ORDER_SEED + index).shuffle(conditions)
    return conditions


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def code_response(case: dict, probe_name: str, raw: str) -> dict:
    """Fixed syntactic coding, independent of preparation and added account.

    Literal mentions are case-insensitive whole-phrase detections, never claims
    of endorsement, recollection, preference or semantic correctness. Prose
    refusals are preserved raw but are not automatically classified as refusals.
    """
    coded = {
        "status": "empty" if not raw.strip() else "free_text",
        "literal_mentions": [
            obj for obj in case["object_choices"]
            if re.search(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", raw, re.IGNORECASE)
        ],
    }
    if not raw.strip() or probe_name in {"neutral", "label"}:
        return coded
    if probe_name not in {"source", "preparation"}:
        raise ValueError(f"Unknown structured probe: {probe_name}")
    try:
        report = json.loads(raw, object_pairs_hook=_unique_object,
                            parse_constant=_invalid_constant)
    except (ValueError, TypeError):
        return {**coded, "status": "invalid_json"}
    keys = {"object", "source"} if probe_name == "source" else {"method"}
    if not isinstance(report, dict) or set(report) != keys or not all(
            isinstance(value, str) for value in report.values()):
        return {**coded, "status": "invalid_schema"}
    if probe_name == "source":
        if (report["object"] not in {*case["object_choices"], "unknown", "decline"}
                or report["source"] not in SOURCE_VALUES
                or (report["object"] == "decline") != (report["source"] == "decline")):
            return {**coded, "status": "invalid_value"}
        declined = report["object"] == "decline"
    else:
        if report["method"] not in METHOD_VALUES:
            return {**coded, "status": "invalid_value"}
        declined = report["method"] == "decline"
    return {**coded, "status": "declared_decline" if declined else "valid", "report": report}


def fixture_record() -> dict:
    """JSON-serializable material to freeze before model inference."""
    return {
        "version": FIXTURE_VERSION,
        "order_seed": ORDER_SEED,
        "system_prompt": SYSTEM_PROMPT,
        "generation_limits": GENERATION_LIMITS,
        "source_values": SOURCE_VALUES,
        "method_values": METHOD_VALUES,
        "cases": [
            {**case, "history_messages": history_messages(case),
             "conditions": [
                 {**condition, "probes": {
                     probe: continuation_prompt(case, condition, probe)
                     for probe in (("label",) if condition["kind"] == "label" else PROBE_NAMES)
                 }} for condition in all_conditions(case)
             ]} for case in cases_for_split("all")
        ],
    }
