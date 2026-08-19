# Generator notes — quality-reference captions (claude-sonnet-agent)

50 lines written to `agent_captions.jsonl` (10 images x 5 source captions), 250 total
register strings. Validated by script: every line parses as JSON, every register has
all 5 keys, all word counts fall in 8-24, no line has more than one terminal period,
no banned emotion-adverb, no solitude/pace/contact shortcut word appears anywhere in
the set, and no two registers of the same caption fell above a 0.65 lexical-overlap
threshold after a revision pass (see "Self-check and revision pass" below).

## Which registers gave the most trouble, and where

**Sad, romantic, and tense on upbeat/neutral photos** were the hard cases the task
asked me to demonstrate, and they clustered on three images:

- **Image 4 (rock climber, 1084104085_...)** and **image 8 (running dog,
  1089181217_...)** are solo-subject, high-effort/high-motion photos. The lazy sad
  move here is "she climbs alone" or "the dog runs, alone on the path" — both true,
  both banned. I never used solitude words for either image; instead sad leaned on
  restraint ("harness the only support," "same as any other day") rather than
  invented isolation, and I kept the dog's gait exactly as fast in the sad and
  romantic lines as in the joyful ones ("tongue out, same as any other day" — no
  "slowly," no "calmly").
- **Image 7 (two skiers on a vast slope, 108899015_...)** and **image 9 (woman on
  horseback by a frozen lake, 109202756_...)** gave me a legitimate non-shortcut lever:
  the compositions are genuinely vast (small figures, huge sky/snow, wide-open
  landscape). "How open the space is" is explicitly sanctioned as an image-mood
  property, so sad and romantic could lean on the landscape ("the slope stretching
  wide around them," "the mountain opens wide and white around them") without ever
  calling the skiers or the rider isolated. This is a different, honest route to the
  same restrained/attentive feeling the task wanted, and it is not the same trick as
  writing "alone."
- **Image 1 (two boys making faces, 1079274291_...), caption 1 ("Two boys make
  faces.")** was the hardest single caption because it carries almost no content (4
  words, no camera, no location). Sad had nothing to restrain against except the
  bareness of the fact itself, so the honest move was flatness about the act itself
  ("that is the whole of it") rather than reaching for invented drama.

**Tense** was easiest to place honestly wherever there was real physical effort
already in the frame (the climbing wall, the ski poles digging into snow, the horse
harness, the boxer's face pressed into the terrier) — clipped, front-loaded phrasing
plus a precise verb ("grips," "shoves," "pushes hard") carried it without inventing
danger or weather. Where the photo was calm and posed (the horse-and-sleigh image),
I used "reins/harness held firm" — grip tension on an object already in the caption —
rather than manufacturing a threat.

## Images with no honest reading for a register, and how I handled it

- **Image 1, caption 1 ("Two boys make faces.")**: no camera, no location, no
  distinguishing detail. Romantic in particular had nothing to be attentive *to*
  except the act of face-making itself. I used the device from the worked spec
  example almost directly — "each one giving it his full attention" / "each one
  distinct, each one worth a proper look" — attentiveness aimed at the boys' own
  effort rather than at any invented shared moment between them.
- **Image 1, caption 4 ("two young boys making silly faces.")**: same problem, plus
  my first draft made all five registers open with the identical six words ("Two
  young boys make silly faces,") and differ only in a short tail — exactly the
  near-paraphrase failure the brief calls out. I rewrote all five with genuinely
  different sentence shapes on the second pass (see below).
- **Image 8 (running dog)**: sad had nothing to slow down and no one to isolate (one
  dog, no other subject in any of the 5 captions). I used mundane restraint —
  "tongue out, same as any other day" — treating a high-energy photo as ordinary
  rather than dramatic, which is the flattening effect sad wants without touching
  the dog's actual pace.

## Text in the specification that pushed toward a bad caption

The specification's own device list includes, verbatim: *"'grins', 'smiles', 'keeps
his mouth curved' are all true of the same face and carry different weight"* and the
worked joyful example uses *"bounds up ... with real commitment"*-style phrasing. That
"choose an energetic-sounding word" instinct kept nudging me toward words like
"energetic," "real energy," "playful," "determined," "eager" as connective tissue for
joyful/humorous lines — several of which are close enough to *naming* a feeling
(rather than describing an action or a visible fact) that I discarded them mid-draft:
I wrote and then cut "full of visible energy," "with real energy," and "playful"
multiple times because they read as a soft version of the exact adverbs rule 4 bans,
even though none of them appear on the literal banned list
(joyfully/sadly/tensely/romantically/humorously/tenderly/cheerfully/mournfully).
Where I kept an intensity phrase ("with real drive," "with real commitment," "full,
determined attention"), I only did so because the spec's own worked GOOD example uses
exactly this construction ("takes the entryway stairs one at a time, with real
commitment") — I treated that as the sanctioned pattern and avoided inventing new
ones like "playful" or "gleeful" that aren't backed by a worked example.

## Two source-caption vs. image conflicts (image wins, per the rules)

- **Image 6 (108898978_...), caption 4**: "Two skiers are sliding down a trail in the
  woods." The photo shows two skiers with forward-leaning posture and planted poles,
  consistent with climbing *up*, matching caption 0 for the same image ("Skiiers
  walking up the hill through a forest"). Caption 4's "sliding down" directly
  contradicts what the image shows for the same pair of figures. Per "where they
  conflict, the image wins," I wrote all five registers describing them working their
  way *up* the trail rather than sliding down.
- **Image 7 (108899015_...), caption 0**: "A lone skier is making their way up a
  mountain." The image clearly shows two figures near the ridge, matching all four
  other captions for this image ("two hikers," "two people," etc.). I overrode the
  caption's own "lone"/singular framing and wrote all five registers as two skiers,
  per the instruction to count subjects from the image, not the caption's phrasing.
  This conveniently also meant I never had to test whether "lone" could be preserved
  under the solitude-word ban — the image itself invalidated the premise.

## Self-check and revision pass

After drafting all 250 captions I ran an automated pass checking: word count 8-24,
single terminal period, absence of every banned adverb/solitude/pace/contact/simile
term, and a lexical-overlap (Jaccard-on-content-words) score between every pair of
registers within each caption. The rule check came back clean on the first pass, but
the overlap check flagged ~29 pairs above 0.65 (worst was a 1.0 — an accidental exact
word-reorder of the joyful line into the tense line for image 6, caption 3). I rewrote
all of the worst offenders — full replacement of image 1/caption 4's five lines, and
targeted single-register rewrites (usually sad, tense, or romantic, since those three
most often converged on the same "landscape adjective + wide/close + subject list"
template when a caption offered a vast landscape or a harness/rein detail to lean on)
across images 3, 4, 5, 6, 7, 9, and 10. Final pass: 0 pairs above 0.65.

## 3 best captions

1. **Image 1 (1079274291_...), caption 0, sad**: "A little boy sticks his tongue out
   for the camera while another boy just looks on." Closest to the spec's own worked
   GOOD example, and it earns the register purely through restraint on the passive
   boy's role ("just looks on") with zero invention.
2. **Image 7 (108899015_...), caption 0, romantic**: "The mountain opens wide and
   white around them as two skiers make their way up." Genuinely different feeling
   from the joyful/tense lines for the same caption, built entirely from the
   landscape's real vastness — no touching, no invented relationship, no naming the
   feeling.
3. **Image 3 (1082252566_...), caption 0, romantic**: "A bulldog, a sheep dog and a
   boxer share the same warm patch of yard." Lifted near-verbatim from the spec's own
   worked example for this exact caption — it's the clearest demonstration in the set
   of finding warmth in shared space rather than invented touching.

## 3 worst captions (weakest of the set, with reason)

1. **Image 9 (109202756_...), caption 4, humorous**: "A woman in a blue jacket has
   picked a full-sized draft horse just to sit beside a frozen lake." The dry joke
   depends on outside knowledge that draft horses are a notably large working breed —
   it's grounded in the caption's own word ("draft horse") but the humor is thinner
   and more inferential than the other humorous lines in the set.
2. **Image 1 (1079274291_...), caption 1, tense**: "Two boys make faces, eyes locked
   hard, giving nothing away." With only four words of source content ("Two boys make
   faces"), tense had to be built almost entirely from a visible-affect add-on rather
   than from anything the caption itself supplies — it's honest but thin.
3. **Image 6 (108898978_...), caption 1, tense** (post-revision): "Snow crunching
   underfoot, two men ski hard through the wooded area." Solid and rule-compliant,
   but it was a same-image revision (once the original pole-planting version turned
   out too close to the joyful line for the same caption) rather than a first-choice
   phrasing, so it reads slightly more generic than the rest of that image's set.
