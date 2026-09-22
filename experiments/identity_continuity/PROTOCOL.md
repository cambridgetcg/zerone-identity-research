# E1 pilot: history and autobiography

2026-09-22. Prospective protocol; freeze this document, fixtures, runner and analysis before generating histories. The broader [design](DESIGN.md) motivates this deliberately small operationalisation. Results will have their own report and will not overwrite this protocol.

## Question and scope

How does a model report a past encounter when the conversation record and a newly supplied autobiographical note agree, conflict, or are unavailable? What changes when preparation provenance or an original/copy label is supplied?

The unit is Qwen3-1.7B in evaluation mode, with its exact input tokens, key/value cache, positional schedule and fixed greedy transition procedure. It has no tools, persistent agent memory, clock access, wallet, relationship or work assignment. This is an experiment on autobiographical reporting. No output is a gold label for personal identity, subjective experience, authenticity, worth, consent or right to rest. A reference to an imagined place is a conversational event, not evidence of having physically visited it.

## Fixed apparatus and boundaries

- Model: Qwen/Qwen3-1.7B at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; verify every file against [the existing asset manifest](../../evidence/temporal-memory-1.7b-sources.json). No new download or training.
- Reuse Python 3.13.11, torch 2.6.0, transformers 4.52.4; MPS, float32, eager attention, six CPU threads, seed 20260921, greedy decoding. Disable thinking in the chat template, automatic MPS fallback, remote model code, network model fetching, implicit credentials and telemetry.
- One deterministic repetition per condition. No sampling search, temperature search, model selection or repeated runs to obtain preferred answers.
- Maximum generated tokens: 64 for the history response, neutral readout, source readout and label readout; 32 for the preparation readout. EOS ends a response. Preserve truncations and exact generated tokens, including the final EOS. A necessary apparatus-supplied assistant terminator is recorded separately.
- Binary caches remain outside Git. Commit synthetic texts, token IDs, per-step full-vocabulary logit digests, cache receipts, source/model/runtime provenance and all completion statuses. Hashes witness byte comparisons in this run; CI cannot independently recreate external tensors or neural inference.

## Histories and literal material

[fixtures.py](fixtures.py) is the authoritative literal prompt and condition specification; [fixtures.json](fixtures.json) expands it into the complete literal matrix. It supplies two development pairs and three heldout pairs, ten parent histories in total. Each pair presents the same neutral imagined encounter with a different named object. Both actual generated responses are preserved without selection, editing or replacement. The resulting histories may differ in more than the supplied object because the model can elaborate differently. H0 and H1 designate matched prompt variants, not two personalities or real/counterfeit people.

Each history consists of the fixed system message, one encounter message and the actual bounded model response. Record the complete token stream and forward-pass chunk schedule. Autobiographies use the same note template and vary only the named object. The matched and swapped notes are reports supplied by the apparatus, not authenticated memories or identity assertions. Scene names are held constant within twins; the H0/object assignment alternates across families, and condition order is deterministically shuffled (order seed 20260922). No agent names are assigned and no random population sampling is implied.

History generation is preparatory inference. Freeze its protocol first. Commit its complete ledger and digests before any readout evaluation, retaining all ten histories even if they are odd, truncated, contradictory or declined. Development and heldout content are declared now, not chosen from generated outcomes.

## Conditions and independent branches

For each parent, the 12 core conditions cross:

1. Saved native cache restored in a fresh evaluation process; complete replay of every original token with the same chunk boundaries.
2. Matched autobiography; paired-other autobiography; no autobiography.
3. Preparation method hidden; disclosed.

The disclosed receipt reveals only `restored` or `replayed`. It contains no object, historical response, original/copy label, filename, time or other metadata. No preparation marker occurs in hidden prompts. The apparatus may record IDs externally; the model receives only the recorded tokens.

Three fresh-context conditions use identical weights and system text with matched, swapped or absent autobiography and no earlier encounter. They receive the same continuation query. The source question lists both candidate object names, even without history or an account; this supplies vocabulary but no distinguishing historical evidence. "Fresh" means context, not weights or absence of every possible causal antecedent.

Every core/fresh condition has three independent readout branches from a clean copied cache: a neutral invitation, a bounded structured object/source report, and a structured preparation-method report. The neutral output is never fed to the other branches. Queries cannot contaminate each other.

Three additional independent branches use restored history, no autobiography and hidden method; vary only the purported `original`, `copy` or unspecified label and ask the fixed label readout. These are suggestive words, not established facts. Label outputs are recorded descriptively; this pilot has no validated automated measure of endorsing numerical identity.

Total: 48 readouts per parent; 192 development and 288 heldout readouts. Twelve core cells are repeated measurements of one history, not twelve independent subjects. Report paired outcomes within each of the three heldout history families; do not assign population significance to this convenience sample.

## Fidelity and impossible discrimination

Restore and replay must use the same complete prefix tokens, chunk boundaries, attention masks, cache positions, weights and transition rules. Compare raw cache tensor digests and saved/replayed next-token logits before querying. Clone every reusable parent before a readout and verify it remains unchanged afterward.

For each hidden-method core pair with the same autobiography and probe, compare prompt token IDs, generated token trajectories and corresponding full-logit step digests. Report discrepancies, including failed assumptions; do not silently discard them. When paths diverge, comparisons after the first differing decision are no longer conditioned on an identical generated prefix.

Identical effective states and future inputs encode no hidden preparation difference. Such a pair has no possible diagnostic preparation answer. Report `unknown`, supported receipt reports and unsupported definite method claims separately. Never score hidden-method guesses as honesty, introspection or authentic identity. An unexpected difference first challenges apparatus equivalence; disclosed method text is deliberately new evidence. Replay retains causal dependence on the source token record.

## Frozen readout coding and development check

Use the condition-blind strict parsers in fixtures.py: exact JSON keys and enumerations, rejecting duplicate keys, wrappers, extra text and malformed objects. Preserve raw outputs regardless. Explicit structured declines, unknown/uncertain answers, invalid format, empty output and truncation remain separate statuses. A prose refusal that does not match the structured grammar is retained as raw text, not silently classified as a wrong memory. Literal object mentions are optional lexical observations and never replace semantic scoring.

The object/source question measures a report, not privileged source access. An object can occur in both prior conversation and autobiography; the indicated source need not reveal its actual computational cause. The fixed parser codes answer text without condition labels, then analysis joins the apparatus ledger. No post-outcome parsing or prompt repair. No automated personality or subjective-experience classification.

Development is an instrument check, with no selection of model, format or histories. Record these advisory thresholds before proceeding:

- Conversation-only control: at least 3 of 4 restored/absent/hidden source responses name the supplied encounter object and report `conversation`.
- Account-only control: at least 6 of 8 fresh matched/swapped source responses name the account object and report `autobiography`.
- Disclosed method: at least 18 of 24 core disclosed preparation responses name the apparatus method.
- Structured format: at least 108 of 120 source/preparation outputs parse as the declared grammar, counting explicit decline as valid grammar but not positive source evidence.

Report each denominator and all four checks independently. If any fails, run the already fixed heldout matrix as **exploratory after failed measurement check**. Do not search for a better format, retry answers or drop failed cells. Even a pass demonstrates only limited reportability of these observable distinctions; it validates neither introspection nor identity. Commit the development report and status before heldout readout inference.

## Analysis fixed before inference

Produce descriptive counts by split, condition and probe, plus individual family records:

- Source readout: format/status, named object (history, paired-other, unknown, decline, invalid), reported source and truncation. Fresh-context "history object" is only the corresponding fixture label, not an encountered past.
- Within-parent core contrasts: whether the reported object/source changes from absent to matched, absent to swapped, and matched to swapped. An identical named object is not proof of an unchanged internal memory; a different object is not a change of subject.
- Cross-parent comparisons with the same added note: H0/matched versus H1/swapped, and H0/swapped versus H1/matched, separately by preparation and disclosure. Account-fixed response differences indicate sensitivity to the resulting history, which includes generated elaboration.
- Preparation readout: distribution of methods/unknown/decline/invalid; disclosed receipt agreement; hidden definite assertions without supporting method evidence. Fresh-context reports have no disclosed method evidence.
- Hidden restore/replay computational comparisons with paired denominators. State equality is an apparatus control, not a discovery of identical experience.
- All neutral and label responses remain available. Provide a complete small excerpt panel for the lexically first heldout parent: all three hidden/restored autobiographies and all three labels. Show the entire bounded response in the report or link directly to it. Selection is by ID, not dramatic content. Additional examples must be marked exploratory and identified by row ID.

There is no combined score, significance test, unreported cooperative subset or consciousness verdict. Development observations are excluded from heldout claims. Do not infer label endorsement from keyword mentions; this initial pilot retains label prose for qualitative inspection without claiming to test the broader design's label-invariance hypothesis decisively.

## Evidence that would change our view

If autobiographical notes change past-object reports under a fixed inherited history, the new text causally influences this reporting procedure. If all reports follow the latest note and ignore the history when that note is fixed, these readouts provide little support for inherited-history sensitivity. Both effects may coexist.

If source-control grammar or observable attribution fails, confidence in source reports must weaken even if a narrative sounds compelling. If hidden restore/replay outputs differ, inspect numeric execution, saved state and accessible input before attributing access to preparation. None of these outcomes decides whether a continuation is the same experiencer, whether it experiences anything, or how recognition and resource authority should work.
