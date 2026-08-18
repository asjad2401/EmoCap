# Stage-02 Configuration Probe

- models: `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` (reference `gemini-3.5-flash`)
- A arms: 30 train images -> 150 calls, 750 captions
- B arms: 60 train images -> 60 calls, 1500 captions (more images because per-position stats need them)
- full run: **A = 40,455 calls**, **B = 8,091 calls** (same 202,275 target captions)
- word target: 8-24; `max_attempts=1` so cost shown is base cost

## Per-arm results

| arm | calls | complete (95% CI) | reject (95% CI) | trunc | in tok | out tok | lat s | words | in-target | recall | novel | reg-sim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-text@gemini-3.5-flash` | 150 | 99% [98-100] | 1% [0-1] | 0 | 710.9 | 120.3 | 3.5 | 15.97 | 100% | 0.8652 | 0.4161 | 0.4367 |
| `A-224@gemini-3.5-flash` | 150 | 100% [100-100] | 1% [0-2] | 0 | 1922.2 | 130.4 | 2.48 | 18.05 | 100% | 0.8204 | 0.5312 | 0.3373 |
| `A-384@gemini-3.5-flash` | 150 | 99% [98-100] | 1% [0-1] | 0 | 1922.7 | 133.7 | 4.41 | 18.49 | 100% | 0.8304 | 0.5325 | 0.3408 |
| `A-512@gemini-3.5-flash` | 150 | 100% [100-100] | 1% [0-3] | 0 | 1920.8 | 130.7 | 5.53 | 18.09 | 100% | 0.8285 | 0.5271 | 0.3446 |
| `B-384-schema@gemini-3.5-flash` | 60 | 100% [100-100] | 2% [1-4] | 0 | 2053 | 654.4 | 6.48 | 16.9 | 100% | 0.9004 | 0.4631 | 0.4049 |
| `B-384-noschema@gemini-3.5-flash` | 60 | 83% [73-92] | 2% [1-3] | 0 | 2053 | 692.9 | 4.94 | 16.43 | 100% | 0.7541 | 0.376 | 0.418 |
| `A-384@gemini-3.6-flash` | 150 | 0% [0-0] | - | 0 | - | - | - | - | 0% | 0.0 | 0.0 | None |
| `A-384@gemini-3.1-flash-lite` | 150 | 100% [100-100] | 1% [0-2] | 0 | 1922.7 | 128.2 | 4.62 | 16.71 | 100% | 0.7352 | 0.5326 | 0.345 |

`recall` = fraction of source content words kept. `novel` = fraction of generated content words absent from the source. `reg-sim` = mean pairwise similarity of the five registers (lower is more distinguishable).

## Rejections by rule

- `A-text@gemini-3.5-flash`: not generated (5), figurative/abstract phrasing (4), 7 words (1)
- `A-224@gemini-3.5-flash`: figurative/abstract phrasing (6), 25 words (1), 26 words (1)
- `A-384@gemini-3.5-flash`: not generated (5), figurative/abstract phrasing (4), 25 words (1)
- `A-512@gemini-3.5-flash`: figurative/abstract phrasing (9), 25 words (1)
- `B-384-schema@gemini-3.5-flash`: figurative/abstract phrasing (29), 25 words (2), 26 words (2), names the emotion with "tensely" (1), 27 words (1)
- `B-384-noschema@gemini-3.5-flash`: not generated (250), figurative/abstract phrasing (19), empty (1)
- `A-384@gemini-3.6-flash`: not generated (750)
- `A-384@gemini-3.1-flash-lite`: figurative/abstract phrasing (7)

## Option B: quality by output position

The specific failure mode for a 25-field response is decay toward the end.

**`B-384-schema@gemini-3.5-flash`** (60 images, so 60 observations per position)

| caption index | returned /5 (95% CI) | rejected /5 |
|---|---|---|
| 0 | 5.00 [5.00-5.00] | 0.07 |
| 1 | 5.00 [5.00-5.00] | 0.07 |
| 2 | 5.00 [5.00-5.00] | 0.17 |
| 3 | 5.00 [5.00-5.00] | 0.17 |
| 4 | 5.00 [5.00-5.00] | 0.12 |

- first-vs-last position difference: 0.000 [0.0, 0.0] -> **no detectable decay**

**`B-384-noschema@gemini-3.5-flash`** (60 images, so 60 observations per position)

| caption index | returned /5 (95% CI) | rejected /5 |
|---|---|---|
| 0 | 4.17 [3.67-4.58] | 0.88 |
| 1 | 4.17 [3.67-4.58] | 0.85 |
| 2 | 4.15 [3.63-4.58] | 0.95 |
| 3 | 4.17 [3.67-4.58] | 0.90 |
| 4 | 4.17 [3.67-4.58] | 0.92 |

- first-vs-last position difference: 0.000 [0.0, 0.0] -> **no detectable decay**

## Paired comparisons

Same images, same captions, same registers -- so these isolate one factor.

**A-text@gemini-3.5-flash** vs **A-384@gemini-3.5-flash** — _does the image change the output at all? equivalence here means multimodal is not earning its 5x input cost_

- paired captions: 745 over 30 images
- identical strings: 0.0%
- mean content similarity: 0.4304 [0.4095, 0.4515]
- **verdict: materially different** (CI entirely below 0.85) -> choose on quality, not cost

**A-224@gemini-3.5-flash** vs **A-384@gemini-3.5-flash** — _does resolution matter?_

- paired captions: 745 over 30 images
- identical strings: 0.1%
- mean content similarity: 0.4515 [0.4325, 0.4704]
- **verdict: materially different** (CI entirely below 0.85) -> choose on quality, not cost

**A-384@gemini-3.5-flash** vs **A-512@gemini-3.5-flash** — _does more resolution matter?_

- paired captions: 745 over 30 images
- identical strings: 0.3%
- mean content similarity: 0.4706 [0.4504, 0.492]
- **verdict: materially different** (CI entirely below 0.85) -> choose on quality, not cost

**A-384@gemini-3.5-flash** vs **B-384-schema@gemini-3.5-flash** — _does batching change caption content?_

- paired captions: 745 over 30 images
- identical strings: 0.0%
- mean content similarity: 0.4356 [0.4119, 0.4603]
- **verdict: materially different** (CI entirely below 0.85) -> choose on quality, not cost

**B-384-schema@gemini-3.5-flash** vs **B-384-noschema@gemini-3.5-flash** — _does the schema change content, or only reliability?_

- paired captions: 1249 over 50 images
- identical strings: 0.6%
- mean content similarity: 0.5019 [0.4807, 0.5243]
- **verdict: materially different** (CI entirely below 0.85) -> choose on quality, not cost

- **A-384@gemini-3.5-flash** vs **A-384@gemini-3.6-flash**: no overlapping captions

**A-384@gemini-3.5-flash** vs **A-384@gemini-3.1-flash-lite** — _does gemini-3.1-flash-lite produce different captions than gemini-3.5-flash?_

- paired captions: 745 over 30 images
- identical strings: 0.0%
- mean content similarity: 0.3841 [0.3655, 0.4033]
- **verdict: materially different** (CI entirely below 0.85) -> choose on quality, not cost

## Cost extrapolation

| arm | full-run calls | input tok | output tok | cost |
|---|---|---|---|---|
| `A-text@gemini-3.5-flash` | 40,455 | 28.8M | 4.9M | _no prices given_ |
| `A-224@gemini-3.5-flash` | 40,455 | 77.8M | 5.3M | _no prices given_ |
| `A-384@gemini-3.5-flash` | 40,455 | 77.8M | 5.4M | _no prices given_ |
| `A-512@gemini-3.5-flash` | 40,455 | 77.7M | 5.3M | _no prices given_ |
| `B-384-schema@gemini-3.5-flash` | 8,091 | 16.6M | 5.3M | _no prices given_ |
| `B-384-noschema@gemini-3.5-flash` | 8,091 | 16.6M | 5.6M | _no prices given_ |
| `A-384@gemini-3.1-flash-lite` | 40,455 | 77.8M | 5.2M | _no prices given_ |

Rerun with `--in-price` / `--out-price` (USD per 1M tokens) for dollar figures. Rates are deliberately not hardcoded.


## Sample output

**`A-text@gemini-3.5-flash`** — source: _A child in a pink dress is climbing up a set of stairs in an entry way ._

- `joyful`: A child in a bright pink dress bounces up the stairs in the entryway.
- `sad`: A single child in a pink dress climbs the entryway stairs alone, step by step.
- `tense`: A child in a pink dress climbs up the entryway stairs, gripping each step.
- `romantic`: A child in a soft pink dress ascends the entryway stairs in a quiet moment.
- `humorous`: A child in a pink dress tackles the entryway stairs, treating them like a mountain.

  novel words vs source: alone, ascends, bounces, bright, climbs, each, entryway, gripping, like, moment, mountain, quiet, single, soft, step, tackles, them, treating

**`A-224@gemini-3.5-flash`** — source: _A child in a pink dress is climbing up a set of stairs in an entry way ._

- `joyful`: A child in a bright pink dress bounces up the stairs into the entryway, eager to explore.
- `sad`: A child in a simple pink dress climbs the worn entryway stairs, alone in the quiet space.
- `tense`: A child in a pink dress grips the entryway frame, climbing the narrow steps with careful, deliberate focus.
- `romantic`: A child in a soft pink dress ascends the entryway stairs, gently touching the wooden frame as she goes.
- `humorous`: A child in a pink dress scales the entryway stairs, apparently on a very serious mission indeed.

  novel words vs source: alone, apparently, ascends, bounces, bright, careful, climbs, deliberate, eager, entryway, explore, focus, frame, gently, goes, grips, indeed, mission
