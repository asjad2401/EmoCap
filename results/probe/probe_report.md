# Stage-02 Configuration Probe

- models: `gemini-3.5-flash` (reference `gemini-3.5-flash`)
- A arms: 6 train images -> 30 calls, 150 captions
- B arms: 12 train images -> 12 calls, 300 captions (more images because per-position stats need them)
- full run: **A = 40,455 calls**, **B = 8,091 calls** (same 202,275 target captions)
- word target: 8-24; `max_attempts=1` so cost shown is base cost

## Per-arm results

| arm | calls | complete (95% CI) | reject (95% CI) | trunc | in tok | out tok | lat s | words | in-target | recall | novel | reg-sim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-384@gemini-3.5-flash` | 30 | 97% [90-100] | 1% [0-3] | 0 | 1921.4 | 132.6 | 3.03 | 18.41 | 100% | 0.8063 | 0.5186 | 0.3299 |

`recall` = fraction of source content words kept. `novel` = fraction of generated content words absent from the source. `reg-sim` = mean pairwise similarity of the five registers (lower is more distinguishable).

## Rejections by rule

- `A-384@gemini-3.5-flash`: not generated (5), figurative/abstract phrasing (2)

## Option B: quality by output position

The specific failure mode for a 25-field response is decay toward the end.

_no B arms run_

## Paired comparisons

Same images, same captions, same registers -- so these isolate one factor.

## Cost extrapolation

| arm | full-run calls | input tok | output tok | cost |
|---|---|---|---|---|
| `A-384@gemini-3.5-flash` | 40,455 | 77.7M | 5.4M | _no prices given_ |

Rerun with `--in-price` / `--out-price` (USD per 1M tokens) for dollar figures. Rates are deliberately not hardcoded.


## Sample output

**`A-384@gemini-3.5-flash`** — source: _A child in a pink dress is climbing up a set of stairs in an entry way ._

- `joyful`: A young child in a bright pink dress bounds up the entryway stairs, eager to see what lies ahead.
- `sad`: A solitary child in a pink dress climbs the worn entryway stairs, moving slowly and quietly all alone.
- `tense`: A child in a pink dress grips the entryway frame, climbing up the stairs with a tight hold.
- `romantic`: A child in a soft pink dress ascends the entryway stairs, gently touching the wooden doorframe as she climbs.
- `humorous`: A child in a pink dress scales the entryway stairs, seemingly on a very serious mission for someone so small.

  novel words vs source: ahead, all, alone, ascends, bounds, bright, climbs, doorframe, eager, entryway, frame, gently, grips, hold, lies, mission, moving, quietly
