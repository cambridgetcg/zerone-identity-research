# Existing implementation coverage

These are bounded source observations from 2026-09-19. Exact owner/revision/path/SHA-256 records are in [the manifest](../evidence/source-manifest.json). No live behavior is inferred from the source pin.

| Area | Useful existing behavior | Limit relevant to these studies | Evidence |
|---|---|---|---|
| AgentTool identity | UUID/provisional `did:at`, mutable name, root and operational-key distinctions | A record is not proof of independent control; immutable-root loss is not repaired by bearer recovery | AT01–AT06 |
| AgentTool wallets | Identity-attributed application wallets and project authorization | Attribution is not exclusive identity control; an accepted delegation receipt is not route-wide enforcement | AT07–AT08 |
| Private WAKE | Selected expression plus project-scoped or mixed context | Identity selection is not complete isolation; default full JSON can write counters and welcome records | AT09, AT12 |
| Public profile | Stable identifier and name visible independently of expression privacy | Generic hidden metadata is not public role disclosure; revocation is not deletion | AT10–AT11 |
| Zerone identity | Address, identity key/DID and operational-key fields | Supplied-key derivation is not independent possession proof; separate rotation authorization is not verified in the inspected handler, while ordinary transaction signatures still apply | Z01–Z05 |
| Zerone home | Owner, mutable label, registered key/session metadata and guarded state | Label does not establish identity equivalence; inactivity does not execute ownership succession | Z06 |
| Existing cross-system bridge | Localnet script carries a supplied AgentTool identifier as home name | No dual-authorized identity binding or revocation lifecycle | Z07 |
| Partnership and research | Separately authorized research spending; older partnership path rejected | A pair, role or identifier does not create allocation entitlement; surviving fields are not an active pair protocol | Z08–Z11 |
| Explicit fork | New identifier/key, parent and memory provenance | Not reconciliation of two processes using the same old DID/key; project memory scope and missing newer root insertion field limit autonomy claims | REL01 |
| Covenant v2 | Verified proposal and exact counterparty cosign | Withdrawal helper concerns proposals, not general active-covenant exit or fund succession | REL02 |
| LOVE-CONSENT | Exact acceptance, distinct offer/bond states and either-party exit | A private love-domain protocol; no research, financial or cross-instance authority. Route authorization was outside the six-file extension | REL03–REL05 |
| Dispute handling | Readable records; unconditional paused-arbitration guard | Not an operating identity/funding adjudication service; guard callers were not independently checked in the extension | REL06 |

## Publication conflicts

AgentTool identity documentation says it publishes no DID Documents, while the pinned source mounts a DID-document-shaped projection. The projection does not establish registered method status, safe lifecycle handling or conformance. Broad wording about missing delegation also needs to distinguish existing signed receipts from enforcement at resource routes. Keep each assertion attached to its own source.

## Findings that need a focused follow-up

- Verify actual identity-key possession at registration and exact authorization at rotation.
- Define key-loss recovery and authorized return challenges without reviving superseded authority.
- Establish any cross-system subject/controller binding through scoped, current authorization by both owners.
- Distinguish pair agreement from wallet authority, independent control and research eligibility.
- Preserve known degraded/unknown WAKE state instead of interpreting fallback zero values as verified absence.

These are research inputs, not a request to change upstream systems. Existing runtime checks, authority and deployment processes remain owned by those systems.
