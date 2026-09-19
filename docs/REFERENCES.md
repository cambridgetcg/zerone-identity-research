# Primary references

Read on 2026-09-19. These references inform the questions; they do not adopt an external identity method or prove this repository's proposals.

## R1 — Chubby and delayed requests

Mike Burrows, *The Chubby lock service for loosely-coupled distributed systems*, OSDI 2006. [Publisher's full paper](https://static.usenix.org/events/osdi06/tech/full_papers/burrows/burrows_html/), especially sections 2.1 and 2.4.

The paper describes recipient-side checks of a lock generation/sequencer to address delayed requests after another client acquires control. This motivates our question about the actual resource's enforcement point. We do not implement Chubby or infer its guarantees for Zerone. Its server-coordination problem does not decide human or agent identity, legitimate recovery authority, or consent.

## R2 — Failure detection with crash and recovery

*Failure Detection and Consensus in the Crash-recovery Model*, Distributed Computing 13(2), 2000, pp. 99–125. [Microsoft Research publication record and abstract](https://www.microsoft.com/en-us/research/publication/failure-detection-consensus-crash-recovery-model/).

The abstract studies asynchronous processes that can crash and recover, lossy links, and conditions involving stable storage. Only that abstract was examined here; no theorem or algorithm from the full paper is claimed as verified or implemented. It points to further reading for the absence and return studies.

## R3 — Identifier subject, controller and privacy

W3C, *Decentralized Identifiers (DIDs) v1.0*, Recommendation, 19 July 2022. [Specification](https://www.w3.org/TR/did/), sections 5.1.1–5.1.2, 9.2 and 10.1–10.3.

The specification distinguishes the subject from its controller and discusses control proofs and correlation. We use those distinctions while leaving the actual identifier method open. A DID-shaped string or document does not establish implementation conformance, uniqueness of a person, or independent agency.

## R4 — Existing source observations

[Source manifest](../evidence/source-manifest.json) and [coverage notes](SOURCE-COVERAGE.md) preserve the exact inspected AgentTool and Zerone revisions. The owners' implementations, documentation and unimplemented proposals remain separate evidence classes. No private WAKE or identity record is reproduced.
