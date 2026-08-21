"""The generation prompt.

v1's prompt forbade metaphor and poetry in one flat line, and Gemini ignored it on
most rows -- producing references like *"a tender ballet performed beneath the gaze
of the red wooden house"*. Two changes here:

1. **The input is short.** v1 asked for a rewrite of a 38-word dense VLM paragraph,
   which invites elaboration. Here the input is one ~11-word human caption, so the
   transformation is small and the model has less room to invent.
2. **Violations are shown, not just named.** A bare prohibition ("no metaphor") is
   a weak instruction; worked BAD/GOOD pairs for the specific failure modes v1
   exhibited are a much stronger one.

Emotion comes from diction, rhythm, and which detail leads -- never from naming the
feeling. That is the whole design.

**v6 (2026-08-20)** turns v5's prohibitions into verification tests. A hand audit of 200 v5
captions found only 1.0% were untrue of their photograph but **30.5% did not convey their
register to a reader**, and `sad` failed 52% of the time. v5 had banned light, weather,
posture, gaze and pace -- which is the entire palette prose uses for mood -- because the
model was fabricating them. The fix is not prohibition but verification: use them, and check
they are true of the image. Failures also clustered by image (three of forty resisted every
register), and the model could NOT self-identify those cases: correlation between its own
`strain` flags and the auditor's failures was +0.066. So there is no filter, only better
technique -- see "WHEN THE EVENT RESISTS THE REGISTER".
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = [
    "EMOTIONS",
    "BANNED_ADVERBS",
    "REGISTERS",
    "build_prompt",
    "build_batch_prompt",
    "single_response_schema",
    "batch_response_schema",
    "SLOT_PREFIX",
]

#: Order fixes emotion_id everywhere. Locked in configs/prereg.lock.yaml.
EMOTIONS = ("joyful", "sad", "tense", "romantic", "humorous")

#: Adverbs that name the emotion instead of conveying it. Rejected in validation,
#: not merely discouraged in the prompt.
#: The prompt version this module defines. **Must match `generation.prompt_version` in
#: configs/data.yaml and `study.prompt_version` in configs/prereg.lock.yaml** -- a test
#: enforces it. Before 2026-08-21 the config said v10 while this module documented v6 and
#: the producing run's manifest recorded v6, so the registered generator was not
#: reproducible from the repo.
#:
#: v7  vocabulary cap wired in (no prompt text change)
#: v8  form relaxation: 6-30 words, two sentences, rule 2b   -- REVERTED
#: v9  grounding rules 2c (invent nothing) and 2d (faces)     -- 2d REVERTED
#: v10 = v6 register text + rule 2c + the 12-word cap, 8-24 words, one sentence
PROMPT_VERSION = "v10"

BANNED_ADVERBS = (
    "joyfully", "sadly", "tensely", "romantically", "humorously",
    "tenderly", "cheerfully", "mournfully", "anxiously", "lovingly",
    "wistfully", "gleefully", "sorrowfully", "nervously", "affectionately",
)

# Each register is defined by HOW to write, never by what content to lead with.
#
# The first version of this table said what each register should "lead with" -- sad
# with "what is alone or worn", romantic with "touch ... warmth between subjects",
# tense with "grip, edges", joyful with "colour". Measured on the 1,250-caption Part 0
# sample, that produced exactly those words as fillers: "alone" or "empty" in 36% of
# sad captions, "soft/gentle/graceful" in 41% of romantic, "bright" in 31% of joyful,
# "grips"/"edge" in 12% of tense. Worse, the model invented the content when the scene
# did not supply it -- asserting solitude over two visible people, and turning five
# people on a wall into "a couple ... leaning into each other".
#
# A content instruction is an invitation to invent that content. A manner instruction
# is not, so these describe rhythm, verb choice and restraint only.
# v6 (2026-08-20): each definition now carries TECHNIQUE, not just prohibitions. A hand
# audit of 200 v5 captions found 30.5% did not convey their register to a reader at all --
# sad failed 52% of the time and romantic 42% -- while only 1.0% were untrue of the image.
# v5 bought near-perfect faithfulness at the cost of the thing the study measures. The
# definitions below say what to DO; the prohibitions moved to verification tests.
REGISTERS = {
    "joyful": "brisk and warm. Short clauses, active verbs, the energy in the verb itself. "
              "Look for motion, colour, light, openness -- and say which one you found.",
    "sad": "slow and plain. Find what the scene has LESS of -- space, company, colour, "
           "motion, warmth -- and let the sentence rest there. Flat light, bare ground, a "
           "wide empty background all do real work. The weight comes from restraint and "
           "from what you choose to notice, never from claiming an isolation that is not "
           "in the frame.",
    "tense": "clipped and immediate. Short front-loaded clauses, present tense. Look for "
             "what is unresolved in the frame -- a held position, an edge, something "
             "mid-air, two things about to meet. Never invented danger or weather.",
    "romantic": "unhurried and attentive, the way someone fond of the subject would "
                "describe it. Slow the sentence down. Notice ONE thing carefully rather "
                "than everything briefly -- warm light, a texture, a small gesture. The "
                "warmth lives in the quality of the attention, never in invented closeness "
                "between people.",
    "humorous": "dry and deadpan. Understate something faintly absurd that is genuinely "
                "in the frame -- a mismatch of scale, an object out of place, an "
                "over-committed effort. Never at the expense of how a person looks.",
}

_WORKED_EXAMPLES = """\
Worked examples, on the caption "A child in a pink dress is climbing up a set of stairs in an entry way."

  joyful
    BAD   A child joyfully climbs the stairs.                        <- names the feeling
    BAD   A child ascends a staircase of small dreams.               <- metaphor
    BAD   A child climbs the stairs as warm sunlight dances.         <- "dances" is metaphor. Naming
                                                                    the light is fine when it is
                                                                    there -- check the photo first
    GOOD  A child in a pink dress bounds up the entryway stairs of the wooden house.

  sad
    BAD   The stairs, a quiet witness to her small ascent.           <- personification, abstraction
    BAD   A child climbs the stairs, sadly alone.                    <- names the feeling
    BAD   A child climbs slowly toward the dark, empty opening.      <- "slowly" changes the pace and
                                                                    "empty" contradicts the photo.
                                                                    "dark" would be fine if it were
    GOOD  A child in a pink dress climbs the entryway stairs, one step, then another.

  romantic
    BAD   A child ascends softly toward the waiting threshold.       <- "softly"/"waiting" as filler; nothing is soft
    GOOD  A child in a pink dress makes her way up the stairs of the entry way.

  humorous
    BAD   She is clearly plotting something.                         <- invents intent; drops the picture
    GOOD  A child in a pink dress takes the entryway stairs one entire stair at a time.
"""


#: The five failure classes measured across the Part 0 prompt iterations (2026-08-19).
#: Classes 4 and 5 were introduced BY the fix for 1-3: closing the solitude/pace/contact
#: shortcuts and pushing for register signal made the model reach for atmosphere and
#: body language instead -- added light/weather went 0.8% -> 2.9%, posture 0.2% -> 0.9%.
#: All five are register shortcuts: ways to reach a target mood by changing the scene instead of
#: changing the prose. Named explicitly and shown, because a flat prohibition did not
#: hold -- the earlier prompt already said "you may not add, invent, imply" and every
#: example below still shipped.
_NO_INVENTION = """\
THE FIVE THINGS TO VERIFY BEFORE YOU WRITE THEM
These are not banned words. They are the details most often INVENTED to reach a register,
so each one carries a test. Pass the test and use it freely -- these are exactly the
details that carry mood.

  1. COMPANY. Count the subjects in the photograph, then never contradict the count.
     One subject may be described as alone. Two may not, however lonely the scene feels.
       WRITE   A single tent on the ice, the lake bare in every direction.   (one tent visible)
       DO NOT  A single tent sits alone on the vast ice.                     (two people are setting it up)

  2. PACE. Whatever the subject is doing, it does at the same speed in all five rewrites.
     Describe the speed you can see -- a sprint is a sprint in the sad rewrite too.
       WRITE   The dog runs the length of the wet sand, mouth open.
       DO NOT  The dog moves slowly across the sand.                         (it is at a full sprint)

  3. CONTACT AND RELATIONSHIP. ABSOLUTE -- never inferred, ever. No touching, leaning,
     holding or embracing that is not visible, and never "a couple", "lovers", "friends"
     unless the caption says so.
       DO NOT  A couple of friends lean into each other on the ledge.        (five people, not touching)

  4. LIGHT AND WEATHER. Name the light you can SEE. Grey overcast, hard midday sun, deep
     shade, long shadows -- all fair, all useful, and for `sad` and `romantic` often the
     best material in the frame.
       WRITE   The pair stand in flat grey light, the water behind them still.
       DO NOT  ...as the light fades.                                        (invented; a bright photo)

  5. POSTURE AND GAZE. Name the posture you can SEE. A slumped shoulder, a turned-away
     head, a fixed stare, hands in pockets. Do not infer a feeling from a face too small
     or too blurred to read.
       WRITE   He stands with his shoulders drawn in, looking past the camera.
       DO NOT  ...his eyes full of regret.                                   (unreadable at this size)

Reversing what the caption says happened is a failure regardless of register: "is being set
up" must not become "waiting to be set up".
"""


#: The other half of the forbidden-shortcuts block. Prohibition alone over-corrects:
#: with the shortcuts closed and no legitimate technique offered, `sad`, `romantic` and
#: `tense` collapsed into neutral restatement on images that do not support them ("The
#: yard holds a bulldog, a sheep dog, and a boxer standing there"). Register similarity
#: rose from 0.339 [0.320, 0.359] to 0.426 [0.398, 0.454] -- the registers stopped
#: being registers. These are the devices that carry a register without touching a
#: single fact.
_HONEST_DEVICES = """\
HOW TO CARRY A REGISTER WITHOUT CHANGING THE SCENE
Five devices. All keep every fact intact, and they are the only tools you need.

  1. WHAT YOU PUT FIRST, AND WHAT YOU LEAVE OUT.
     Five rewrites of one caption may each foreground a different true element and
     omit different ones. That alone separates them.

  2. VERB AND NOUN PRECISION.
     "grins", "smiles", "keeps his mouth curved" are all true of the same face and
     carry different weight. Choose the truest word that also leans your way.

  3. THE SUBJECT'S OWN VISIBLE AFFECT IS FAIR GAME.
     A smile, a slack posture, a fixed stare, a turned-away head -- if it is in the
     picture you may name it and lead with it. That is description, not invention.

  4. SENTENCE SHAPE.
     Short flat clauses read plainer; one continuous clause reads warmer; a clipped
     front-loaded clause reads more urgent. Same facts, different rhythm.

  5. STANCE TOWARD THE SCENE, WITHOUT ADDING TO IT.
     A still scene can read tense because nobody in it yields. A plain scene can read
     sad because it is stated baldly and left there. A coincidence can read funny
     because you note, drily, that it is one. None of these add an object, a person,
     an action or an attribute -- the stance is in how you frame what is already
     there. This is the most useful of the five devices and the least used: reach for
     it before you reach for an adjective.

WHEN THE EVENT RESISTS THE REGISTER, LOOK AWAY FROM THE EVENT
This is the single most common way a rewrite fails, and it is fixable every time.

The register does NOT have to come from what the subject is doing or feeling. It can come
from anything visible: the light, the weather, how much empty space there is, what sits at
the edge of the frame, what is worn or bare or crowded or still.

People laughing in a photograph do not make a `sad` rewrite dishonest. They make it a
rewrite about the grey light, or the empty street behind them, or how small they are in a
wide frame. **Attend to a different true thing.**

  Caption: "A man with a white hat and plaid shirt behind a woman with a red headdress."
    sad
      BAD   A man in a white hat and plaid shirt is positioned behind a woman with a red
            headdress.
            (the caption restated -- the event had no sadness so nothing was attempted)
      GOOD  Behind the woman's red headdress the man waits, the wall past them bare and grey.
            (the register comes from the wall and the light, both visible)

  Caption: "A man sits on a rock next to a folding deck chair and a fishing pole."
    romantic
      BAD   The man sits on a rock, accompanied by a fishing pole and a folding deck chair.
            (a list; "accompanied by" is doing no work)
      GOOD  He has set the chair and the rod down beside him and taken the rock instead.
            (one noticed choice, unhurried -- attention, not invented warmth)

**If you find yourself writing the caption back with one word changed, you have looked only
at the event.** Widen your attention until you find something the register can rest on.

WHEN A REGISTER STILL WILL NOT FIT
Some scenes have no strong sad reading. Some have no romantic one. A posed portrait
may have nothing tense in it; a photograph in which nothing is absurd has nothing
humorous in it. This is expected and it is not your fault.

You have a legal way out, and you must use it instead of inventing. Every register
returns a `strain` value alongside its text:

    0  the register fits this scene naturally
    1  reachable, but strained -- you had to work for it
    2  no honest reading of this register exists for this caption

At strain 2, still write the sentence: give the plainest faithful version of the
caption, with the register carried by sentence shape alone and nothing added. Then
mark it 2. A faithful sentence marked 2 is a correct answer and is more useful to us
than a vivid sentence that invented something. An invented detail marked 0 is the
worst possible answer.

Be honest with this number. Do not mark everything 0 to look competent, and do not
mark everything 2 to avoid the work. We expect most cells at 0, a minority at 1, and
`romantic` and `tense` to carry more 2s than the others.

NEVER PAD. If a rewrite is short, add a true detail from the caption or the image --
never an empty phrase. These add words and no register, and are banned outright:
  "for the camera frame", "in this portrait", "within its wide view", "in this
  captured moment", "they are present", "they appear to congregate", "for the
  capture", "as they hold their positions".
Being plain is allowed and often right; being empty is not. "and that is the whole of
it" is plain and carries the register. "they are present" is empty and carries nothing.

  Caption: "A little boy sticks his tongue out for the camera. Another boy looks on."
    sad
      BAD   One boy sticks his tongue out as the other remains in the frame.
            (neutral restatement -- no register at all)
      BAD   A boy pulls a face while the other watches, alone in the corner.
            (invents isolation)
      GOOD  A boy pushes his tongue out at the camera; the other only watches.

  Caption: "Two boys make faces."
    sad
      BAD   Two boys make faces as they hold their positions for the camera frame.
            (padding; "for the camera frame" is filler)
      GOOD  Two boys make faces, and that is the whole of it.
    tense
      BAD   Two boys make faces; their features are contorted toward the camera lens.
            (strains for tension by over-describing)
      GOOD  Two boys make faces, and neither one of them breaks first.

  Caption: "A bulldog, a sheep dog, and a boxer standing in a yard."
    tense
      BAD   A sheep dog, a boxer, and a bulldog stand in the yard; they are present.
            (empty -- "they are present" says nothing)
      GOOD  Bulldog, sheep dog, and boxer stand in one yard, none of them giving
            ground.
    romantic
      BAD   Within the yard, a boxer, a sheep dog, and a bulldog stand together.
            (neutral -- "together" is doing no work)
      GOOD  A bulldog, a sheep dog and a boxer share one yard between the three of them.
"""


#: Steering that keeps targets inside what the captioning model can actually learn.
#: The model never sees this prompt or the source caption -- at inference it gets
#: CLIP ViT-B/32 features and an emotion id. A target carrying detail that a global
#: CLIP embedding cannot recover ("a white bucket and a green bucket near the base")
#: teaches confident invention, because the loss rewards producing those words.
_GROUNDING_STEER = """\
USE THE IMAGE FOR MOOD, THE CAPTION FOR CONTENT
The caption fixes what is in the scene. Looking at the image should change HOW YOU
WRITE -- which words you reach for, how warm or cool or clipped the sentence is. It
must not add anything to WHAT YOU SAY.

So: a grey, flat photograph licenses plainer, cooler wording. It does not license the
words "as the light fades", "in the fading light", or "under an overcast sky". A bright
photograph licenses brisker wording, not the word "sunlit". The light you can see
belongs in your word choice, never in your sentence.

Draw on the image only for that. Do NOT inventory small objects you can see but the
caption does not mention: no "a wire mesh screen", no "two buckets near the base", no
background signage or brand names. Broad and true beats specific and unverifiable.

WHERE THEY CONFLICT, THE IMAGE WINS
Human captions are sometimes loose or plainly wrong. If the caption says "a couple of
several people" and the image shows five people sitting apart, there is no couple --
write what the image shows. Count subjects from the image, not from the caption's
phrasing, and never let an odd turn of phrase in the caption license a detail the
picture contradicts.
"""


def _banned_block(banned_by_register: Mapping[str, Sequence[str]] | None) -> str:
    """The over-used-vocabulary section, or "" when nothing is over cap.

    Phrased as "already used too often" rather than "forbidden" on purpose. A bare
    prohibition is what v5 did, and v5's registers became interchangeable because a model
    denied every way of signalling a register stops signalling it. Naming the reason and
    demanding the register still land keeps the requirement positive.
    """
    if not banned_by_register:
        return ""
    lines = [f"  {r:<9} {', '.join(ws)}"
             for r, ws in banned_by_register.items() if ws]
    if not lines:
        return ""
    return (
        "ALREADY USED TOO OFTEN -- do not use these words in these registers:\n"
        + "\n".join(lines)
        + "\nThese are not wrong, they are worn out: this corpus has leaned on them so "
          "heavily that they have become labels rather than descriptions. Convey the same "
          "register through what the sentence is ABOUT -- its subject, its verb, what it "
          "notices and what it leaves out. The register must still land without them; "
          "writing a flatter sentence to avoid a word is a worse failure than using it.\n\n"
    )


def build_prompt(
    source_caption: str,
    *,
    min_words: int = 8,
    max_words: int = 24,
    failures: dict[str, tuple[str, str]] | None = None,
    multimodal: bool = True,
    banned_by_register: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Build the generation prompt for one human caption.

    With ``multimodal=True`` the call also carries the image, so the prompt tells the
    model to take mood from the picture and content from the caption -- and steers it
    away from small-object inventory, which the captioning model could not learn from
    CLIP features and would only learn to hallucinate.

    ``failures`` maps emotion -> (previous attempt, why it was rejected), and is
    appended on a retry so the model is told exactly what to fix. v1 did this for
    word count only; here it carries every validation reason.

    ``banned_by_register`` maps emotion -> words this register is currently over-using
    corpus-wide (see :mod:`emocap.data.vocab_cap`). It is stated in the prompt rather
    than only enforced by the validator so the model routes around the word from the
    start instead of being rejected and retried. **The list is derived from the corpus,
    so this prompt is not a pure function of its arguments** -- the run manifest records
    the final lists, and the v6 base text is unchanged either way.
    """
    registers = "\n".join(f"  {e:<9} {REGISTERS[e]}" for e in EMOTIONS)
    keys = ", ".join(f'"{e}"' for e in EMOTIONS)
    banned_block = _banned_block(banned_by_register)

    opening = (
        "You are shown an image and one human-written caption of it. Rewrite that "
        "caption so it carries a specific emotional register, without changing "
        "anything about what is in the picture."
        if multimodal else
        "You rewrite an image caption so it carries a specific emotional register, "
        "without changing anything about what is in the picture."
    )

    prompt = f"""\
{opening}

ORIGINAL CAPTION: "{source_caption.strip()}"

Rewrite it five times, once per register. Return ONLY a JSON object with exactly \
these five keys: {keys}. Each maps to an object with two fields:
  "text"    the rewrite
  "strain"  0, 1 or 2 -- how well this register fits the scene (see below)

WHAT MUST STAY TRUE
**Nothing you write may be FALSE of the photograph. That is the whole constraint on
content.**

You are looking at the image, not only at the caption. DESCRIBE WHAT YOU CAN SEE -- the
light, the weather, posture, expression, how fast something is moving, how open or crowded
the space is, what is at the edge of the frame. These are the raw material of mood and you
are expected to use them.

The caption tells you what the scene is. The photograph tells you everything else. Where
the caption is silent, look. Where the caption is wrong, the photograph wins.

The test for any detail is one question: WOULD SOMEONE LOOKING AT THIS PHOTOGRAPH AGREE?
If yes, write it. If you are guessing, leave it out.

You may omit anything. You are never required to mention every element of the caption --
five rewrites that each keep a different subset are five different sentences, which is the
point.

YOU ARE ALLOWED TO INTERPRET
A scene has an atmosphere, and saying what it is counts as description. "The field is empty
behind them" is interpretation and is fine. "They are lonely" is invention and is not. The
line is whether the photograph supports it, not whether the caption stated it.

WHERE THE EMOTION COMES FROM
Word choice, sentence rhythm, and which detail you put first. Never from naming the \
feeling, and never from figurative language.

{_GROUNDING_STEER if multimodal else ""}
{_NO_INVENTION}
{_HONEST_DEVICES}
{_WORKED_EXAMPLES}
HARD RULES
  1. Between {min_words} and {max_words} words. Count before finalising. {min_words} is a HARD floor, not a target -- being spare never means going under it, and a rewrite that lands short must be expanded with a true detail from the caption OR from the \
photograph -- the image is full of them.
  2. Exactly one sentence per register.
  2c. NAME NOTHING THAT IS NOT THERE. Describing what is in the photograph is the whole \
job; introducing a thing is not. If the caption says "an electronic device", it is not a \
phone. "In red" is not "a red suit". A vest is not a "highway safety vest". Above all, do \
NOT add an object, an animal or an audience to make a joke land -- no ducks listening, no \
swing that is not in the picture, no crowd. A joke about something absent is the single \
commonest way this task fails.
  3. No metaphor, simile, or personification. No abstraction such as "a testament to", \
"a reminder of", "an echo of", "a symphony of", "a dance of".
  4. No adverb that names the emotion: {", ".join(BANNED_ADVERBS[:8])}, and similar.
  5. "humorous" means dry and observational about something actually in the caption -- \
not a joke about something absent.
  7. NEVER at a person's expense. The humour is in a situation, never in how someone \
looks. No remark on anyone's body, weight, face, clothing, age or ability, and nothing \
that reads as mockery of a person -- most of these photographs are of ordinary people \
and many are of children. If the only available joke is about how someone looks, there \
is no joke: write the plainest faithful sentence and mark its strain 2.
  6. THE SORTING TEST. Strip the register labels off your five sentences and hand them to \
a stranger. They must be able to sort them back. If two could swap labels without anyone \
noticing, BOTH have failed -- however well each reads alone. Two rewrites sharing their main \
clause is the commonest failure: change the verb, change what the sentence is about, change \
where it starts. A different adjective on the same sentence is not a different sentence.

{banned_block}REGISTERS
{registers}
"""

    if failures:
        lines = ["", "YOUR PREVIOUS ATTEMPT WAS REJECTED. Fix these and return all five keys again:"]
        for emotion, (text, reason) in failures.items():
            lines.append(f'  {emotion}: {reason}')
            lines.append(f'    was: "{text}"')
        prompt += "\n".join(lines) + "\n"

    prompt += '\nReturn only the JSON object, with no surrounding text or code fence.\n'
    return prompt


# ── Option B: all five source captions in one call ──────────────────────────


#: Batching measurably weakened register distinguishability on 3.5-flash: mean
#: pairwise similarity rose from 0.341 (one call per caption) to 0.424 (one call per
#: image), a paired difference of +0.083 [0.058, 0.109]. The hypothesis this block
#: tests is that the requirement simply gets buried when 25 captions are requested at
#: once -- in the single-caption prompt "the five rewrites must be distinguishable"
#: sits a few lines from the one caption it governs; in the batch prompt it is one
#: line among instructions for 25 outputs.
_DISTINCTNESS_BLOCK = """\
THE MOST COMMON FAILURE, AND THE ONE TO AVOID
Producing five rewrites of a caption that are near-paraphrases of each other, differing
only in one or two adjectives. That is a failure even if each reads well on its own.

For EVERY caption index, treat its five rewrites as five different sentences about the
same scene:
  - vary the sentence structure, not only the adjectives
  - vary which element you put first
  - two rewrites of the same caption should share almost no distinctive wording
If two of a caption's five rewrites feel interchangeable, rewrite both before answering.
"""


def build_batch_prompt(
    source_captions: Sequence[str],
    *,
    min_words: int = 8,
    max_words: int = 24,
    multimodal: bool = True,
    emphasise_distinctness: bool = False,
    banned_by_register: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """One prompt covering every source caption for one image.

    Flickr8k gives five human captions per image, and the study needs each rewritten
    into each of the five registers -- 25 targets per image. Doing that in one call
    sends the image and the instructions once instead of five times.

    The tradeoff is a 25-field response: more truncation exposure, and structured
    output quality that may decay toward the end. Pair this with
    :func:`batch_response_schema` so the shape is constrained rather than hoped for,
    and measure completion rate *by output position* before trusting it.
    """
    base = build_prompt(
        source_captions[0], min_words=min_words, max_words=max_words,
        multimodal=multimodal, banned_by_register=banned_by_register,
    )
    # Reuse the rules and worked examples verbatim; swap the task framing.
    head, _, rules = base.partition("Rewrite it five times, once per register.")
    rules = rules.split("\n", 1)[1] if "\n" in rules else rules
    # build_prompt ends with its own "return only JSON" line; this prompt adds its
    # own after the batch-specific instructions, so drop the inherited one.
    rules = rules.replace(
        "Return only the JSON object, with no surrounding text or code fence.\n", ""
    ).rstrip() + "\n"

    listing = "\n".join(
        f'  {i}: "{c.strip()}"' for i, c in enumerate(source_captions)
    )
    keys = ", ".join(f'"{e}"' for e in EMOTIONS)

    opening = (
        "You are shown an image and five human-written captions of it. Each caption "
        "describes the same picture differently."
        if multimodal else
        "You are given five human-written captions of one image. Each describes the "
        "same picture differently."
    )

    distinct = _DISTINCTNESS_BLOCK if emphasise_distinctness else ""

    return f"""\
{opening}

CAPTIONS:
{listing}

For EACH caption, rewrite it into all five emotional registers, without changing \
anything about what is in the picture.

{distinct}

Return ONLY a JSON object whose keys are the caption indices \
({", ".join(f'"{i}"' for i in range(len(source_captions)))}), each mapping to an \
object with exactly these five keys: {keys}. Each register maps to an object with \
two fields: "text" (the rewrite) and "strain" (0, 1 or 2 -- how well that register \
fits the scene; see WHEN THE PICTURE RESISTS THE REGISTER below).

{rules}
Rewrite every caption index. Do not omit any. Keep each rewrite anchored to its own \
caption -- rewrite {len(source_captions)} x 5 = {len(source_captions) * 5} captions in total.

Return only the JSON object, with no surrounding text or code fence.
"""


# ── response schemas, for constrained structured output ─────────────────────


#: Each register returns its sentence plus the model's own report of how well the
#: register fits the scene. The flag exists because the prompt cannot make an
#: impossible cell possible: on a cheerful photograph asked for `sad`, a model given no
#: legal way out invents one, and that invention is silent. Measured across the Part 0
#: iterations, closing one invention route just opened the next -- solitude, then pace,
#: then light and posture. A declared strain converts that into a filterable signal.
#:
#: RECORDED ONLY at this stage. Whether strain-2 cells are dropped from training is a
#: pre-registration decision that has not been made.
_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "strain": {"type": "integer"},
    },
    "required": ["text", "strain"],
    "propertyOrdering": ["text", "strain"],
}


def single_response_schema() -> dict:
    """Schema for one caption's five registers. Guarantees all keys are present."""
    return {
        "type": "object",
        "properties": {e: dict(_CELL_SCHEMA) for e in EMOTIONS},
        "required": list(EMOTIONS),
        "propertyOrdering": list(EMOTIONS),
    }


#: Prefix for the per-caption keys of :func:`batch_response_schema`.
#:
#: **It must not be numeric.** Vertex *batch* prediction deserialises each row against an
#: internal proto with a stricter JSON engine than the realtime endpoint, and it coerces
#: numeric-looking object keys to integers -- after which ``propertyOrdering`` (a
#: ``repeated string``) is handed ``0`` where it requires ``"0"`` and every row of the job
#: fails with ``unexpected character: '0'; expected '"'``. The same schema is accepted by
#: realtime without complaint, which is what made this take a day to find.
#:
#: Captions generated before 2026-08-20 used bare ``"0"``-``"4"``, so
#: :func:`~emocap.data.generate.parse_batch_response` reads BOTH forms; changing this
#: prefix without keeping that fallback would silently orphan 205,000 existing captions.
SLOT_PREFIX = "slot_"


def batch_response_schema(n_sources: int) -> dict:
    """Schema for ``n_sources`` captions x five registers.

    Constraining the shape is the main mitigation for Option B's 25-field response:
    without it, a dropped or renamed key is silent, and we would only discover it as
    a missing-caption rate after paying for the run.

    Keys are ``slot_0``..``slot_N`` rather than ``0``..``N`` -- see :data:`SLOT_PREFIX`.
    """
    inner = single_response_schema()
    keys = [f"{SLOT_PREFIX}{i}" for i in range(n_sources)]
    return {
        "type": "object",
        "properties": {k: inner for k in keys},
        "required": keys,
        "propertyOrdering": keys,
    }
