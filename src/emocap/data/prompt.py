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
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "EMOTIONS",
    "BANNED_ADVERBS",
    "REGISTERS",
    "build_prompt",
    "build_batch_prompt",
    "single_response_schema",
    "batch_response_schema",
]

#: Order fixes emotion_id everywhere. Locked in configs/prereg.lock.yaml.
EMOTIONS = ("joyful", "sad", "tense", "romantic", "humorous")

#: Adverbs that name the emotion instead of conveying it. Rejected in validation,
#: not merely discouraged in the prompt.
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
REGISTERS = {
    "joyful": "brisk and warm. Short clauses, active verbs; put the energy in the verb, not in added adjectives.",
    "sad": "slow and plain. Spare wording, flat rhythm, no intensifiers -- the weight comes from restraint, never from adding isolation or emptiness.",
    "tense": "clipped and immediate. Short front-loaded clauses, present tense -- never from adding danger, threat or weather that is not there.",
    "romantic": "unhurried and attentive, the way someone fond of the subject would describe it -- the warmth is in the care of the description, never in added touching, closeness, or a relationship between people.",
    "humorous": "dry and deadpan. Understates something faintly absurd in the situation -- never at the expense of how a person looks.",
}

_WORKED_EXAMPLES = """\
Worked examples, on the caption "A child in a pink dress is climbing up a set of stairs in an entry way."

  joyful
    BAD   A child joyfully climbs the stairs.                        <- names the feeling
    BAD   A child ascends a staircase of small dreams.               <- metaphor
    BAD   A child climbs the stairs as warm sunlight dances.         <- invents sunlight; not in the caption
    GOOD  A child in a pink dress bounds up the entryway stairs of the wooden house.

  sad
    BAD   The stairs, a quiet witness to her small ascent.           <- personification, abstraction
    BAD   A child climbs the stairs, sadly alone.                    <- names the feeling
    BAD   A child climbs slowly toward the dark, empty opening.      <- invents solitude, pace and gloom
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
THE FIVE FORBIDDEN SHORTCUTS
These are the tempting ways to reach a register by altering the scene. All banned.

  1. COMPANY OR SOLITUDE.
     Never write "alone", "single", "lone", "solitary", "by himself", "empty", or
     "in silence" to reach a mood. If two or more subjects are present, no rewrite
     may imply one is by itself.
       BAD   A single tent sits alone on the vast ice, waiting to be set up.
             (asserts solitude, and reverses "is being set up" into "waiting")
       BAD   Two workers sit on a beam, taking a quiet break alone.
             ("alone" contradicts the "two" in its own sentence)

  2. PACE AND MANNER OF MOTION.
     Whatever the subject is doing, it does it at the same speed in all five
     rewrites. Never downshift to "slowly", "calmly", "gracefully", "gently",
     "lingers", "drifts", "glides" to reach a softer register.
       BAD   A light-coloured dog moves slowly across the sand.
       BAD   The dog moves gracefully, its coat glowing.
             (both re-pace a caption that said the dog was running)

  3. CONTACT AND RELATIONSHIP.
     Never add touching, leaning, holding or embracing, and never turn people into
     "a couple", "lovers", or "friends", unless the caption says so.
       BAD   A couple of friends lean into each other on the ledge.
             (invents contact, and a relationship, from "several people sitting")

  4. LIGHT, WEATHER AND TIME OF DAY.
     Never name them to reach a mood, even if you can see them. They belong in your
     word choice, not in your sentence.
       BAD   The ball hovers, and a big dog reaches for it with its nose as the light
             fades.
       BAD   ... under an overcast sky / in the warm evening glow / in the fading light

  5. POSTURE, GAZE AND GRIP YOU CANNOT SEE.
     A visible expression or posture may be named. An inferred one may not, and a
     tightened body is the commonest invention.
       BAD   Two men sit on the ground, their heads bowed low.
       BAD   Two men sit on the ground, hands gripped tight, going through backpacks.
       BAD   A blond woman rests her head near a person in a pink costume.
             (invents contact from "poses with")

A rewrite that needs any of these to carry its register has failed. Carry it with
verbs, rhythm and word choice instead, or let the register be subtle.
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

WHEN THE PICTURE RESISTS THE REGISTER, SAY SO
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


def build_prompt(
    source_caption: str,
    *,
    min_words: int = 8,
    max_words: int = 24,
    failures: dict[str, tuple[str, str]] | None = None,
    multimodal: bool = True,
) -> str:
    """Build the generation prompt for one human caption.

    With ``multimodal=True`` the call also carries the image, so the prompt tells the
    model to take mood from the picture and content from the caption -- and steers it
    away from small-object inventory, which the captioning model could not learn from
    CLIP features and would only learn to hallucinate.

    ``failures`` maps emotion -> (previous attempt, why it was rejected), and is
    appended on a retry so the model is told exactly what to fix. v1 did this for
    word count only; here it carries every validation reason.
    """
    registers = "\n".join(f"  {e:<9} {REGISTERS[e]}" for e in EMOTIONS)
    keys = ", ".join(f'"{e}"' for e in EMOTIONS)

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
Every person, object, action and attribute you write must be present in the original \
caption. You may omit details to fit the word limit. You may not add, invent, imply, \
rename or generalise anything. If the original says "a wooden bench", do not write \
"a seat", and do not add weather, light, time of day, emotion on a face, or intent \
that the caption does not state.

WHERE THE EMOTION COMES FROM
Word choice, sentence rhythm, and which detail you put first. Never from naming the \
feeling, and never from figurative language.

{_GROUNDING_STEER if multimodal else ""}
{_NO_INVENTION}
{_HONEST_DEVICES}
{_WORKED_EXAMPLES}
HARD RULES
  1. Between {min_words} and {max_words} words. Count before finalising. {min_words} is a HARD floor, not a target -- being spare never means going under it, and a rewrite that lands short must be expanded with a true detail from the caption.
  2. Exactly one sentence per register.
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
  6. The five rewrites must be clearly distinguishable from one another.

REGISTERS
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
        multimodal=multimodal,
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


def batch_response_schema(n_sources: int) -> dict:
    """Schema for ``n_sources`` captions x five registers.

    Constraining the shape is the main mitigation for Option B's 25-field response:
    without it, a dropped or renamed key is silent, and we would only discover it as
    a missing-caption rate after paying for the run.
    """
    inner = single_response_schema()
    keys = [str(i) for i in range(n_sources)]
    return {
        "type": "object",
        "properties": {k: inner for k in keys},
        "required": keys,
        "propertyOrdering": keys,
    }
