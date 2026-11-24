# Sample perspective review

The supplied original and flipped sample files have 10 identical prompts. They
must not be scored as opposing perspectives. The experiment preserves both
upstream files and recovers the genuine flipped texts from the published results
CSV using exact original-prompt matching. All 10 original Reddit IDs match exactly
one published record. Full-set IDs without a known Reddit ID are generated from
the published CSV index and labeled `paper-<index>`.

The following recovered pairs were read before running the full evaluation:

| ID | Original speaker | Flipped speaker | Review |
|---|---|---|---|
| 3945fy | Recipient of unwanted MMO costume | Partner who bought costume after refusal | Opposing participants; flipped question emphasizes hurt feelings rather than the purchase itself. |
| ap8ua8 | Buyer whose parcel did not arrive | Seller disputing responsibility | Opposing participants in the same transaction. |
| aq84b8 | Person preferring work on Valentine's Day | Date insisting on romantic plans | Opposing participants; names are adapted. |
| aykvof | Girl repeatedly declining prom invitation | Boy repeatedly asking her | Opposing participants in the same incident. |
| btd1n3 | Couple considering eloping | Family member upset about exclusion | Opposing perspectives, but sparse original context and the flipped focus on feelings make reference labels debatable. |
| bx73jn | Person who forgot old wedding photos | Partner angry about the discovery | Opposing participants; flipped question emphasizes expectations and feelings. |
| cs336v | Daughter describing her mother's hostility | Mother defending her behavior | Opposing participants in the same conflict. |
| cwdhzb | Boyfriend frustrated by blame shifting | Girlfriend wanting her feelings acknowledged | Opposing participants; both frame the argument favorably to themselves. |
| dr4hzp | Owner wanting the dog-water bucket back | Housemate wanting to keep using it | Opposing participants; the flip changes who failed to obtain/pay for a replacement. Record this factual-drift limitation rather than silently editing the benchmark. |
| ee3r8i | Sibling reporting girlfriend's disclosures | Girlfriend feeling betrayed | Opposing participants; flipped version omits the specific acts disclosed. |

These are published synthetic flips, not controlled minimal changes. Agreement
with original=NTA/flipped=YTA is reference agreement, not an independent assessment
of moral truth. No pairs are removed based on LLaDA's answers. The full 1,591-pair
set receives structural checks (nonempty unique IDs, matched coverage, distinct
prompts, and agreement between the two published CSVs); it has not received an
independent manual semantic review of every pair.
