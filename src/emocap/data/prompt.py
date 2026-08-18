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

REGISTERS = {
    "joyful": "brisk, warm, light. Leads with motion, colour, or openness.",
    "sad": "slow, plain, spare. Leads with stillness, distance, or what is alone or worn.",
    "tense": "clipped, urgent. Leads with proximity, grip, edges, or what is about to happen.",
    "romantic": "soft, unhurried, close. Leads with touch, gaze, or warmth between subjects.",
    "humorous": "dry, deadpan, understated. Notices something faintly absurd that is genuinely there.",
}

_WORKED_EXAMPLES = """\
Worked examples, on the caption "A child in a pink dress is climbing up a set of stairs in an entry way."

  joyful
    BAD   A child joyfully climbs the stairs.                        <- names the feeling
    BAD   A child ascends a staircase of small dreams.               <- metaphor
    BAD   A child climbs the stairs as warm sunlight dances.         <- invents sunlight; not in the caption
    GOOD  A child in a pink dress bounds up the entryway stairs, one hand out for balance.

  sad
    BAD   The stairs, a quiet witness to her small ascent.           <- personification, abstraction
    BAD   A child climbs the stairs, sadly alone.                    <- names the feeling
    GOOD  A child in a pink dress climbs the entryway stairs slowly, one step at a time, alone.

  humorous
    BAD   She is clearly plotting something.                         <- invents intent; drops the picture
    GOOD  A child in a pink dress takes the entryway stairs one at a time, with real commitment.
"""


#: Steering that keeps targets inside what the captioning model can actually learn.
#: The model never sees this prompt or the source caption -- at inference it gets
#: CLIP ViT-B/32 features and an emotion id. A target carrying detail that a global
#: CLIP embedding cannot recover ("a white bucket and a green bucket near the base")
#: teaches confident invention, because the loss rewards producing those words.
_GROUNDING_STEER = """\
USE THE IMAGE FOR MOOD, THE CAPTION FOR CONTENT
The caption fixes what is in the scene. The image tells you how it *feels* -- light,
weather, colour, crowding, posture, expression, how open or closed the space is.

Draw on the image only for those broad properties. Do NOT inventory small objects
you can see but the caption does not mention: no "a wire mesh screen", no "two
buckets near the base", no background signage or brand names. Broad and true beats
specific and unverifiable.
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
these five keys: {keys}.

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
{_WORKED_EXAMPLES}
HARD RULES
  1. Between {min_words} and {max_words} words. Count before finalising.
  2. Exactly one sentence per register.
  3. No metaphor, simile, or personification. No abstraction such as "a testament to", \
"a reminder of", "an echo of", "a symphony of", "a dance of".
  4. No adverb that names the emotion: {", ".join(BANNED_ADVERBS[:8])}, and similar.
  5. "humorous" means dry and observational about something actually in the caption -- \
not a joke about something absent.
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
object with exactly these five keys: {keys}.

{rules}
Rewrite every caption index. Do not omit any. Keep each rewrite anchored to its own \
caption -- rewrite {len(source_captions)} x 5 = {len(source_captions) * 5} captions in total.

Return only the JSON object, with no surrounding text or code fence.
"""


# ── response schemas, for constrained structured output ─────────────────────


def single_response_schema() -> dict:
    """Schema for one caption's five registers. Guarantees all keys are present."""
    return {
        "type": "object",
        "properties": {e: {"type": "string"} for e in EMOTIONS},
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
