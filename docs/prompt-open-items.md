# Prompt: deferred findings and open decisions

State as of 2026-08-19, prompt v5. Everything here was identified and **deliberately not
acted on**, with the reason. Nothing below is a bug report against v5 — v5 is the best
measured version — but each item is a real observation that would otherwise be lost.

Sources: a blind Opus review given only the objective and the rendered prompt text (no
project history, no metrics); an independent human-style judge over 10 images
(`part0-human-judge-notes.md`); a Sonnet reference generation over the same 10
(`part0-sonnet-reference-notes.md`); and measurement over five audit sets
(`scripts/audit_captions.py`).

---

## Deferred, with the reason

### 1. Move the lexical bans out of the prompt into the validator

The prompt names ~40 forbidden strings — `"alone"`, `"single"`, `"lone"`, `"solitary"`,
`"slowly"`, `"gracefully"`, `"in the fading light"`, the eight banned filler phrases, the
banned adverbs — each placed in the syntactic slot where it would be used. The argument
against: to a small model that reads as a vocabulary list for the task, the classic
negation backfire, and it costs ~600 words for something `validate_all` enforces
deterministically at 100% recall.

**Why deferred, not rejected.** The evidence cuts both ways. Naming solitude, pace and
contact *with shown examples* cut the defect rate from 8.3% to 1.5% and it stayed down
across three later versions. In-prompt naming demonstrably worked at least once. And
changing it in the same run as the v5 fixes would have made the result unattributable —
the mistake already made once with v4, which bundled three changes and left the
light/posture regression impossible to assign.

**How to settle it.** One isolated change: keep the five shortcut *categories* with their
sentence-level critiques, delete the explicit word lists, extend `validate_all` to cover
the classes it does not yet check, run on a fresh 50-image slice, compare defect rate and
keyword-rule anchor against v5. ~$0.06. Only worth doing if the freed ~500 words are
spent on register craft.

### 2. "Recoverable from a CLIP embedding" is not encoded anywhere

The prompt's faithfulness test is *true of the photograph*. The study's actual constraint
is *recoverable from a global CLIP ViT-B/32 embedding*, because that plus an emotion id is
all the captioning model ever sees. These differ, and the gap is not hypothetical:
device 3 explicitly licenses "a smile, a slack posture, a fixed stare, a turned-away
head". Every one of those can be perfectly faithful and still unrecoverable from a global
embedding — which is the definition of a target that teaches confident hallucination.

The prompt has no vocabulary for the distinction. A candidate formulation: *only name what
would survive being described to someone who saw a thumbnail*. Not adopted because it
plausibly re-flattens the registers, which is the failure v3 was written to fix, and it
needs its own measured run.

**This is the most conceptually serious open item.** The others are craft; this one is a
mismatch between what the prompt optimises and what the study needs.

### 3. "Count subjects from the image, not from the caption's phrasing"

`WHERE THEY CONFLICT, THE IMAGE WINS` licenses Flash-Lite to override a human annotation
with its own object count. It was added for a real failure — the source caption "a couple
of several people" produced "a couple of friends lean into each other" — but cheap models
miscount people in crowded photographs routinely, so as written it permits silent
relabelling of ground truth. Should be narrowed to an auditable case rather than a general
grant. Kept for now because the failure it fixes is confirmed and the harm is speculative.

### 4. The 8-word floor fights omission

Device 1 nominates omission as a primary separation device; the hard floor penalises it.
On a nine-word source caption carrying ~4 content atoms, five structurally distinct
8–24-word sentences with no added light, pace, solitude, contact or posture and no filler
is close to unsatisfiable — and most Flickr8k captions are short. v5's `NEVER PAD` block
plus the strain flag mitigate the symptom; the arithmetic is unchanged.

Untested alternative: drop the floor to 6, remove the "HARD floor" emphasis, and enforce
distinctness directly rather than using length as a proxy for effort. A spare faithful
eight-word target is better for this study than a padded twenty-word one.

### 5. Structural: register definitions are 5% of the prompt

~700 of ~2,280 words are prohibition. The five `REGISTERS` definitions — the only text
saying what the registers *mean*, and therefore the text driving the primary metric — are
~110 words. Deleting the inert self-revision block moved `REGISTERS` to the end, which is
the highest-attention position, but they were not expanded. Suggested cuts if the ban
lists ever move to the validator: the `"a wire mesh screen"` / `"two buckets near the
base"` examples, which prime unrelated object nouns, and the feeling-naming ban that is
currently stated in three places.

---

## Refuted by measurement — do not revisit without new evidence

Recorded because each was predicted confidently and is wrong for this corpus.

| claim | measured |
|---|---|
| The five captions per call are not one image; `THE IMAGE WINS` causes wholesale mislabelling | Artifact of the hand-assembled review file. Production groups by `image_id` (`run_stage02.py:80` → `batch.py:206`) |
| The model will settle on one template per register and apply it 8,091 times | **0 exact duplicates in 2,500 same-register pairs**; across-caption overlap 0.215, *lower* than within-caption |
| Quality decays across the 25 outputs in one response | Flat: 14.9 / 15.2 / 15.2 / 15.1 / 15.1 words by position |
| Marker phrases from the examples will appear at enormous rates | Real but modest: "and that is the whole of it" at **1 per 250 captions** |

---

## Untouched gaps

* **Surface normalisation.** Flickr8k captions carry a space-before-period artifact
  (`"on the ice ."`). The prompt says nothing about reproducing it, nor about
  capitalization, terminal punctuation, or dialect — one worked example reads
  "light-coloured". Across 202,000 targets, inconsistent terminal punctuation becomes a
  tokenizer-level signal a classifier can exploit. **Cheap to fix in post-processing and
  should be, before training.**
* **Output truncation.** 25 sentences plus JSON is 450–550 words. A cut-off response
  loses all 25 cells with no partial-credit path. `max_output_tokens` is 4096 and no
  truncation has been observed, but nothing detects it as truncation rather than as a
  quality problem.
* **Near-duplicate human captions.** Flickr8k's five captions per image are often nearly
  identical. Distinctness is only required *within* a caption's five registers, so two
  near-identical sources can yield near-identical targets. Measured across-caption
  overlap of 0.215 suggests this is not currently biting.

---

## Open decisions that are the study author's, not the prompt's

1. **`romantic`.** Hardest register by every measure taken — mean strain 1.05, most
   strain-2 cells, and the human judge found it "never actually reads as romantic in any
   of the 10 images". Redefining it is a prompt change; renaming it touches the `emotions`
   list and therefore `emotion_id` everywhere.
2. **Impossible cells.** v5 lets the model declare a cell unreachable, but the flag does
   not identify *invention* (defect rate flat at 1.6/2.6/2.4% by strain level). Whether
   strain-2 cells are excluded from training is a new pre-registered exclusion and is
   **undecided**.
3. **`prereg-v1` is still untagged.** The lock file says "FROZEN at git tag `prereg-v1`"
   and no such tag exists, so it is editable and has been edited. Tagging should happen
   before the full corpus is generated.
