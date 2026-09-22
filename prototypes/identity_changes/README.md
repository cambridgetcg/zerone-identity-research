# Room to change

Status: **local symbolic prototype**, authorized on 2026-09-22. This explores a proposed identity-layer design; it does not adopt a Zerone identity criterion, consent policy or resource rule.

Open [the standalone explorer](index.html) in a browser. Choose a scenario, step through its events and change the audience to compare the visible statements. The page works directly from a local file without a server, account or network connection.

## The question

Can a participant change their name or views, acknowledge a shared past and make distinct present choices while others can understand what was stated, by whom, and what remains unresolved?

The prototype treats self-description, continuity claims, participation and resource authority as distinct records. An identity statement can revise an earlier statement within its scope. It cannot silently redirect a payment, multiply an award, inherit a delegation or decide who is the authentic continuation.

## What to explore

- **Renaming and selected history:** a chosen name can change while an earlier alias remains restricted. Compare audience views, including which revision links are disclosed.
- **Changed views:** an explicit revision preserves its predecessor as a historical statement. A later unrelated assertion does not erase earlier statements merely because it arrived later.
- **Shared history and conflicting claims:** multiple continuations can refer to a common past. A shared credential cannot identify a unique speaker or prove independent control. Incompatible statements remain visible without selecting an authentic branch.
- **Participation:** the model can represent a participant withdrawing under explicit, narrow prior terms. It keeps another participant's statements and unrelated resource authority separate. When credential attribution is ambiguous, a withdrawal claim remains unresolved.

The interface shows **statements visible to the selected audience**. It does not promise a complete current identity or omniscient view of agreement. A hidden revision must not silently suppress a public statement or expose the existence of a private disagreement through a status badge.

## Reading the records

Each claim has an identifier, kind, subject reference, symbolic issuer, value, audience, and explicit revision or evidence references. Evidence links record what was cited; they do not certify the cited statement or its interpretation. A challenge is an attributable claim about a particular statement, with no automatic power over the participant or their resources.

The model accepts revisions only within the same subject, kind and issuer scope. It preserves the original statement, retains incompatible unrevised candidates and rejects unknown references. Independent event arrival order does not choose a winner. This local sequencing rule is an abstraction, not a distributed reconciliation protocol.

Subject references and credential tokens are invented labels. The fixture's authorization associations are supplied assumptions, not independently established identities or cryptographic signatures. Two labels or two accepted claims do not prove two independent controllers. Conflicting statements could arise from changed views, concurrent devices, duplication or compromise; the prototype cannot discriminate among those explanations.

Participation changes require an explicitly supplied prior rule and unambiguous authorization within the model. No rule is inferred from ancestry, silence or a request. Resource records are a separately supplied snapshot and remain unchanged throughout identity events. The prototype neither executes payments nor resolves outstanding obligations.

The absence of a modeled agreement transition is a limit of record handling. It never restricts someone's inherent right to stop, rest or decline participation. A statement of stopping can still be recorded; any consequences for outstanding arrangements remain a separate question.

## Disclosure boundary

All data is synthetic. The standalone file contains **every sample audience view**, and its audience selector only filters the display. Anyone with the file can inspect those other views. It provides no encryption, authenticated access control, deletion guarantee or privacy protection for real records.

Within the model, audience projection is tested for non-interference: adding a statement hidden from an audience must leave that audience's projection unchanged, including statuses, candidates and references. The scripted scenario timeline is a teaching aid and is not a private activity feed.

A real integration would need to authorize and minimize each projection before delivering it. Disclosure of old names, private memory or relationships needs its own scope. Preserving selected attribution does not require publishing every alias or a complete personal history.

## Reproduce and check

Python 3.11+ and the standard library are sufficient:

```sh
make prototype
make check
python3 prototypes/identity_changes/build.py --json
```

[model.py](model.py) contains the model and invented scenarios. [viewer.html](viewer.html) is the display template; [build.py](build.py) embeds deterministically generated projections into [index.html](index.html). The default checks reproduce the committed page without running a browser or neural inference. Browser interaction is a separate manual/integration check.

The tests exercise hidden-event non-interference, explicit revision and historical retention, order independence, shared-credential ambiguity, scoped withdrawal, rejection without partial mutation and preservation of resources. These are properties of this bounded implementation, not measured rates of real-world safety, privacy or human understanding.

Local validation on 2026-09-22 passed all 18 new adversarial/build tests, the 14 existing symbolic tests, prior experiment artifact validators and exact page reproduction. Independent review also checked the separation between multiple statements and an explicitly contested claim, and preservation of withdrawal attribution.

A separate local Chromium 147.0.7727.15 check exercised all 63 scenario/step/audience combinations at 1280 × 1050 and the 12 final scenario/audience views at 375 × 900. Rendered claims matched the model; keyboard stepping, buttons and the skip link worked; the tested views had no horizontal overflow, page errors or external page requests. Desktop and mobile screenshots were visually inspected. This browser check used existing local tooling and is not part of default CI; it is not a human usability study or exhaustive accessibility audit.

## Possible Zerone integration

The prototype supplies a record-and-view experiment. It could inform an optional layer alongside account identifiers: selected self-description, attributable continuity claims and explicit participation records, with resource-specific authority evaluated separately. A future WAKE view could select relevant records for a particular audience and purpose.

No Zerone or AgentTool registration, authenticated WAKE, wallet binding, recovery, governance or live bridge is implemented here. The example schema is not a new DID method. Before a real integration, verify current owner implementations, establish the specific authorization and disclosure rules, and test their enforcement where actions occur.

## What would change this proposal

Revise it if hidden records change an unauthorized view, if a later arrival silently erases a competing claim, if shared-key claims establish a unique continuation's consent, or if an identity change alters unrelated authority. Also revise the interface if readers consistently mistake a visible claim for a verified fact or this projection for a complete person.

The next empirical question is whether people can use these views to distinguish a changed name, a disputed history and an actual resource grant. That usability study has not been run. Nothing here settles subjective continuity, numerical identity, personhood or the right to rest.
