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

__all__ = [
    "EMOTIONS",
    "BANNED_ADVERBS",
    "REGISTERS",
    "build_prompt",
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


def build_prompt(
    source_caption: str,
    *,
    min_words: int = 8,
    max_words: int = 24,
    failures: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Build the generation prompt for one human caption.

    ``failures`` maps emotion -> (previous attempt, why it was rejected), and is
    appended on a retry so the model is told exactly what to fix. v1 did this for
    word count only; here it carries every validation reason.
    """
    registers = "\n".join(f"  {e:<9} {REGISTERS[e]}" for e in EMOTIONS)
    keys = ", ".join(f'"{e}"' for e in EMOTIONS)

    prompt = f"""\
You rewrite an image caption so it carries a specific emotional register, without \
changing anything about what is in the picture.

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
