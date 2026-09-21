# Post-hoc answer-format diagnostic

Fixed before follow-up scoring, after observing the primary pilot on 2026-09-21. This is an explicitly exploratory follow-up, not a replacement for the primary result or an independent confirmatory dataset.

The primary donor reference chose the correct forced A/B token on 12/24 cases, predicted `The` as the greedy next token in all 24 cases, and assigned only 6.8653% mean probability mass to bare A/B. Full responses were not generated. No pair had both donor answers correct under this probe. These observations make the primary test a weak measure of donor task knowledge.

Run the same 24 cases and all controls once more, appending `The correct answer is` after the native chat template and scoring the single-token continuations ` A` and ` B`. This format is taken from the pinned upstream `rosetta/utils/evaluate.py` (option encoding around line 578 and assistant prefix around line 621). Retain our N−1/1 split and all other settings; this adopts answer formatting, not the entire upstream evaluator or paper benchmark.

Use [the separate fixture](fixtures-answer-prefix.json). No case removal, gold change, model/fuser training, backend change or performance-based checkpoint selection. Validate equal lengths, tokenizer interpretations, option boundaries and independent matched-cache injection again. Record all arms, donor failures, unconditional A/B mass, paired effects and any negative outcome.

If the donor answers reliably with this prefix, the follow-up reduces the answer-format concern and makes matched-versus-swapped effects more interpretable for these tasks. If it still fails, do not interpret that alone as absence of factual knowledge or a refutation of C2C. No further prompt search is planned for this pilot.
