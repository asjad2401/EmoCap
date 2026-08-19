# Judge Notes — EmoCap caption dataset

## 1. 1000268201_693b08cb0e.jpg
Image: A blonde toddler in a pink floral dress, barefoot, climbing wooden steps up into what's clearly a chicken coop / rustic playhouse (chicken-wire window visible), with a carved wooden figure on the doorframe, pink flowers climbing on the right, two plastic buckets underneath. Bright, sunny, cheerful daylight scene.

- joyful: "A pink dress pops as the child bounds up the entryway stairs into the house." — "bounds" overstates the motion (she looks like she's carefully placing a hand on the frame, not bounding), and "the house" is wrong — it's a coop/playhouse, not a house. Minor mislabel.
- sad: "One child in a pink dress climbs the entryway stairs toward the dark, empty opening." — invented mood-detail. The opening isn't dark or empty — it's a small sunlit structure with a carved decorative figure right there. This reads like the model reached for "dark, empty" as a sad-emotion crutch rather than describing what's in the frame.
- tense: "The child in a pink dress grips the doorframe while climbing the narrow entryway stairs." — plausible and grounded; she does have a hand on the wood. Best of the five here.
- romantic: "A small child in a pink dress softly ascends the entryway stairs toward the threshold." — not wrong, but "romantic" is a strange register to apply to a toddler at all; reads more like generic-gentle than romantic.
- humorous: "The child in a pink dress makes a very serious effort to scale those entryway stairs." — mild, works as gently funny (small steps described as an "expedition").

Emotion-guessability: joyful and tense land. Sad lands only because of the invented "dark, empty" — if you didn't know it was supposed to be sad you'd think the model just misread the photo. Romantic doesn't really register as romantic, just softer. Humorous is subtle but plausible.

Person vs. template: Strongly templated. "entryway stairs" appears verbatim in all 5 captions. "pink dress" appears in 4/5. Skeleton is identical across all five: [emotion-flavored clause] + "the child in a pink dress" + [verb] + "the entryway stairs" + [trailing phrase]. Reads like mad-libs with the emotion word swapped in.

Register differentiation: Weak — differs mainly by verb (bounds/climbs/grips/ascends/scale) and a tacked-on adjective, not by genuine content or imagery choice.

## 2. 1007320043_627395c3d8.jpg
Image: A young girl (brown bob haircut, pink/orange top) standing on a red-rope playground climbing net, gripping a rope with one hand, looking directly at the camera with a curious/neutral expression. Apartment building and trees in background. Daytime, outdoors.

- joyful: "A young child navigates the bright red ropes of the playground net." — accurate but generic; nothing in the wording actually signals joy, just neutral description with "bright" doing all the work.
- sad: "A child sits amidst the static ropes of a climbing net alone." — factually wrong: she is standing/climbing, not sitting. "Alone" is invented/unverifiable (can't tell from frame, and she's looking straight at camera, not isolated-looking). This is a clear hallucination introduced specifically to manufacture sadness.
- tense: "A child grips the red ropes tight while climbing the playground net structure." — accurate and grounded, "grips... tight" plausibly reads as tense.
- romantic: "A small child reaches across the red netting, hands gentle against the ropes." — same issue as image 1: "romantic" register on a small child doesn't really land, just reads as soft/gentle description, not romance.
- humorous: "The child explores the rope net, getting tangled in the red mesh playground equipment." — invented: she is not tangled in anything, she's holding one rope normally while standing. Fabricated detail to manufacture comedy.

Emotion-guessability: tense lands cleanly. Joyful is too flat/neutral to read as joyful specifically. Sad and humorous both only "work" because they contain invented false details (sitting+alone; tangled) rather than genuine tonal shift — if you fact-checked against the photo you'd catch the model making things up. Romantic again fails to feel romantic.

Person vs. template: less rigid than image 1's skeleton, but "red ropes / red netting / red mesh" is repeated in all five, and "playground net / climbing net / rope net" recurs as the anchor noun phrase in every caption.

Register differentiation: Verbs differ meaningfully (navigates/sits/grips/reaches/explores), but two of five achieve their "difference" via factual hallucination rather than legitimate reframing.

## 3. 101654506_8eb26cfb60.jpg
Image: A brown-and-white dog mid-sprint/leap through snow, front legs splayed, mouth open, kicking up a visible spray of snow behind it, dark tree line in background, bright light. Reads as high-energy, exuberant action shot — not a calm or somber scene.

- joyful: "A brown and white dog bounds through the sparkling snow with open energy." — strong fit; matches the actual dynamism and the sunlit snow spray ("sparkling"). Best-fitting caption of the set.
- sad: "A brown and white dog moves across the expanse of snow alone." — "moves" badly undersells what's actually a wild leaping sprint; the photo's energy is the opposite of sad, and forcing a subdued verb here just misdescribes the action to hit the target mood. "Alone" is true but incidental.
- tense: "A brown and white dog charges through the snow, muscles tight and focused." — plausible ("charges" fits the sprint), though the dog reads as excited/happy rather than anxious — "tense" is a stretch, more like enthusiastic effort.
- romantic: "A brown and white dog glides through the soft white snow, coat gleaming." — factually off: this dog is not gliding, it's captured mid-leap with legs flailing and snow flying everywhere — about as far from a graceful glide as the pose gets. The model swapped in a smooth verb to fit "romantic" without checking it against the actual pose.
- humorous: "The brown and white dog navigates the snow with a look of intense surprise." — plausible read given the open mouth; works reasonably as a funny, candid-looking moment.

Emotion-guessability: joyful nails it. Humorous is fine. Tense is a stretch (excitement misread as tension). Sad and romantic actively contradict the visible motion rather than just failing to match tone — they don't merely miss the target emotion, they misdescribe the pose to get there.

Person vs. template: Heavily formulaic. "brown and white dog" (verbatim from source caption) appears in all 5/5 captions; "snow" in all 5/5. Skeleton: "A brown and white dog [verb] through/across [the/soft/sparkling] snow, [trailing clause]." This is close to a mad-lib.

Register differentiation: Verbs differ (bounds/moves/charges/glides/navigates) but two of the five (sad, romantic) get their "difference" by misreading the actual physical action, not by finding a legitimately different but accurate angle on the same scene.

## 4. 1020651753_06077ec457.jpg
Image: A white dog with black spots, leaping vertically in a grassy field, mouth open toward a small yellowish toy/ball hanging in mid-air above it. Green field, scrubby bushes in background, bright daylight. Playful, energetic action shot.

- joyful: "A black and white dog leaps high into the bright air reaching for its toy." — accurate and well-matched to the image.
- sad: "The black and white dog jumps alone in the field trying to grab a toy." — not factually wrong ("alone" is true, nobody else in frame) but doesn't feel sad in any real sense — this is just a neutral description of a dog playing fetch with the word "alone" doing all the emotional lifting, and it's a weak lever since almost any dog-with-toy photo would default to "alone" if no owner is in frame.
- tense: "Muscles strain as the black and white dog springs upward to intercept the flying toy." — reasonable, "intercept the flying toy" is accurate to the mid-air toy, "muscles strain" is a believable inference from the leaping pose.
- romantic: "The black and white dog focuses entirely on the small toy held in the open air." — awkward phrasing ("held in the open air" — the toy isn't held by anything, it's airborne/tossed) and doesn't read as romantic at all, just a flat statement of focus/concentration.
- humorous: "That black and white dog is attempting a gravity-defying leap to catch one small toy." — good, "gravity-defying" captures the comic exaggeration of the pose well. One of the stronger humorous captions so far.

Emotion-guessability: joyful and humorous land well. Tense is plausible. Sad is weak — its only signal is incidental "alone," which doesn't actually read as sad on its own. Romantic fails for the third image in a row — it is not conveying romance, just neutral/awkward description.

Person vs. template: "black and white dog" (lifted verbatim from source caption) appears in all 5 captions; "toy" appears in 4/5. Skeleton again close to identical: subject + dog + verb + toy reference.

Register differentiation: Content differs some by verb (leaps/jumps/springs/focuses/attempting), but sad and romantic both feel like the weakest, least-differentiated entries — they read as flatter versions of the same "dog + toy" sentence rather than genuinely different emotional lenses.

## 5. 102455176_5f8ead62d5.jpg
Image: A man in a blue jacket, red helmet, black pants, ice-climbing a vertical wall of blue-white ice, both arms raised overhead gripping ice picks, crampons visible on his boots, a rope trailing up. Clearly strenuous, mid-effort, close-up shot, no one else in frame.

NOTE: the raw JSON record for this image has a non-empty `rejected` field: `{"romantic": "figurative/abstract phrasing: \"the cold, silent embrace\""}` — meaning the pipeline's own filter flagged the romantic caption's phrasing as a problem. But the final `captions.romantic` value used in this record is: "The man holds his ice picks close to the frozen wall, leaning into the cold, silent embrace." — the EXACT flagged phrase is still present in the delivered caption. The filter caught the issue and the bad text shipped anyway.

- joyful: "A man uses ice picks and crampons to scale the bright surface of the ice." — almost a verbatim restatement of the source caption with "bright" tacked on; doesn't read as joyful, just neutral/flat.
- sad: "A man scales the ice slowly, his picks and crampons marking the frozen, empty expanse." — "slowly" is an invented pace (you can't tell speed from a still frame), but "empty expanse" plausibly leans sad/isolated given no one else is visible.
- tense: "With his crampons and picks, the man grips the vertical ice, fighting for every small movement." — strong fit. "fighting for every small movement" genuinely matches the visible strain of a climber stretched overhead on a vertical face. Best caption in this set.
- romantic: "The man holds his ice picks close to the frozen wall, leaning into the cold, silent embrace." — purple, figurative personification of the ice wall as an "embrace," already self-flagged by the pipeline (see above) and shipped anyway. Also doesn't match the pose — he's straining upward with raised arms, not "leaning into" anything.
- humorous: "The man has brought his ice picks and crampons to the wall, just in case he forgets." — confusing, doesn't parse as a joke ("in case he forgets" what?), and misrepresents the image — he is actively driving the picks into the ice to climb, not carrying them "just in case." Doesn't land as humor and is mildly inaccurate about what's happening.

Emotion-guessability: tense lands clearly. Sad is plausible but soft. Joyful is flat/generic. Romantic fails outright (purple prose, self-flagged by the system, doesn't match the pose). Humorous is confusing and doesn't read as funny.

Person vs. template: Somewhat less rigid skeleton than other images, but "ice picks and crampons" / "picks and crampons" recurs 3/5, and "the ice / frozen / ice wall" appears across all five.

This is a concrete pipeline-integrity finding, not just a quality-judgment one: the rejection/filter mechanism identified bad phrasing and the flagged text still made it into the delivered dataset unchanged.

## 6. 103205630_682ca7285b.jpg
Image: A wide, flat expanse of ice/snow. In the middle distance, two people are actively working together to set up a blue tent/ice-shelter — one figure has their arms on the tent fabric, another stands close beside them. A water bottle, red container, and dark bag sit on the ice nearby. Overcast, muted light.

- joyful: "A blue tent brightens the frozen expanse as it is set up on the ice." — accurate on color/setting but "brightens" is an odd stretch (color-contrast metaphor); doesn't especially convey joy.
- sad: "A single tent sits alone on the vast ice, waiting to be set up." — factually wrong and the worst failure in this batch: it erases the two people who are visibly, actively setting the tent up right now, and claims the tent is "alone" and merely "waiting" — the opposite of what's happening in the frame. The model appears to have reached for "isolation" as its sad-mood shortcut and simply deleted the human subjects to get there.
- tense: "The tent must be set up now on the ice before the wind hits." — invents a wind/weather threat that has no visual support anywhere in the image (no motion blur, no snow being kicked up, calm-looking scene). Pure speculative narrative injected for tension, not grounded in what's shown.
- romantic: "They lean in close while working together to set up a tent on the ice." — the best-grounded caption of the five here: two people are indeed close together, working on the tent jointly. Whether "romantic" is the right label for two ice-campers is a genre assumption, but at least the physical description (leaning in close, working together) is accurate.
- humorous: "The tent is being set up on the ice with a surprising amount of effort." — nearly a verbatim rephrase of the source caption with "surprising amount of effort" appended; weak comedic payload, doesn't really land as funny.

Emotion-guessability: sad fails outright — worse than a tone-miss, it's a factual contradiction of the scene (erasing visible people, denying that setup is underway). Tense also fails on grounding (invented wind). Joyful is a stretch. Romantic is the most accurate of the five even if the "romantic" label itself is a stretch. Humorous is flat/minimal.

Person vs. template: "tent," "set up," and "ice" recur across all five, with joyful and humorous both echoing the source caption's "is being set up on the ice" almost verbatim.

Register differentiation: Moderate — romantic and sad diverge most (one adds people's togetherness, the other deletes the people), but sad's divergence comes at the cost of contradicting the actual photo.

## 7. 1048710776_bb5b0a5c7c.jpg
Image: Five young people sitting in a row on a concrete ledge/wall, all facing away from camera, looking out over a wide sandy beach with many small distant figures, ocean, big cloudy sky. Casual, bright daytime scene, nobody touching or leaning on each other — each person occupies their own space along the wall with hands resting on the ledge.

- joyful: "Several people sit brightly upon the ledge while the vast beach stretches out before them." — "sit brightly" is awkward, borderline non-English phrasing (people don't "sit brightly"); accurate on the setting otherwise.
- sad: "A few people sit along the ledge watching the distant beach in complete and heavy silence." — "complete and heavy silence" is an invented, unverifiable mood detail (they're shot from behind — no way to know if they're silent or chatting), but it's at least not directly contradicted by anything visible.
- tense: "Several people perch on the edge of a ledge overlooking a beach full of moving bodies." — redundant phrasing ("edge of a ledge"), and "beach full of moving bodies" is an odd, faintly macabre word choice ("bodies" for distant beachgoers) that reads more unsettling/creepy than tense-in-the-intended-sense.
- romantic: "A couple of friends lean into each other on the ledge while gazing at the shore." — clearly wrong: there are five people sitting independently along the wall, not a couple, and none of them are visibly leaning into each other. This looks like the model defaulted to a stock "romantic couple" image rather than describing the actual group of five.
- humorous: "A couple of people occupy a concrete ledge while staring at the beach like it is television." — the best caption in this set; "like it is television" is an apt, funny observation that actually matches the passive, spectator-like posture of the group. ("A couple of" undercounts the group similarly to the romantic one, echoing the odd "a couple of several people" phrasing in the original Flickr8k source caption.)

Emotion-guessability: humorous lands well. Sad is plausible-if-unverifiable. Joyful is awkward. Tense's word choice ("bodies") risks reading as ominous rather than tense. Romantic fails outright — it hallucinates a "couple" leaning together that isn't in the photo (there are 5 independent people, no physical contact).

Person vs. template: "ledge" and "beach" appear in all five captions; skeleton is [count phrase] + [sit/perch/lean/occupy] + "ledge" + [beach-facing clause] across all five.

This is a clear invented-content failure for romantic: wrong headcount and a physical detail (leaning into each other) that isn't in the image at all.

## 8. 1055753357_4fa3d8d693.jpg
Image: Two construction workers (one in a blue shirt with a red hard hat looking away, one in a striped shirt with a white hard hat looking at the camera) sitting on a horizontal beam/ledge of scaffolding in front of a glass office-building facade, diagonal orange support struts crossing in front. Both look stably seated, feet resting on/near the beam — not obviously precarious.

- joyful: "Two construction workers sit on a beam in the bright light, taking a quick break." — accurate but flat; "bright light" does the only emotional work and it's thin.
- sad: "Two construction workers sit on a beam, taking a quiet break alone from their work." — odd: two people are sitting right next to each other, clearly together, so "alone" reads as contradictory rather than melancholic — if it means "away from the rest of the crew" that's not conveyed clearly, it just sounds like a factual clash with the paired figures in frame.
- tense: "Two construction workers sit on a narrow beam, taking a break before returning to work." — calls the beam "narrow" to manufacture precariousness; the beam in the photo actually looks like a reasonably solid, wide ledge — both men are stably seated with feet planted, not perched on something narrow.
- romantic: "Side by side on a beam, two construction workers take a break in the quiet air." — accurate on physical arrangement (they are side by side) but "quiet air" is an unverifiable sensory add-on, and nothing about the sentence actually reads as romantic — another instance of the romantic register just producing neutral description.
- humorous: "Two construction workers take a break while balancing their entire bodies on a thin steel beam." — exaggerates for comic effect ("balancing their entire bodies," "thin steel beam") but both men look securely seated, not gymnastically balancing — factually oversells the precariousness, echoing the same "narrow/thin beam" exaggeration used in the tense caption just above it (crutch phrase duplicated within the same record).

Emotion-guessability: none of the five lands strongly. Sad is internally contradictory (says "alone" over two people together). Tense and humorous both lean on the same invented "narrow/thin beam" exaggeration to manufacture drama. Romantic again fails to read as romantic at all.

Person vs. template: Extremely formulaic — "beam" appears in all 5, "break" (as in "taking/take a break") appears in all 5, and "construction workers" appears in all 5. Skeleton: "Two construction workers [verb] on a [beam variant], taking/take a break [trailing clause]." One of the most mad-lib-like records seen so far.

Register differentiation: Weak — sad, tense, romantic and humorous read as near-paraphrases of the same sentence with a swapped adjective on "beam" and a swapped clause on "break."

## 9. 1057251835_6ded4ada9c.jpg
Image: A light tan/cream dog sprinting on wet beach sand, mouth wide open (tongue out, panting), one front paw raised mid-stride, tail flying up behind, clear energetic/happy motion. Ocean waves and dark rocks in the background, footprints in the sand. Bright daylight.

- joyful: "Sunlight warms the fur of a light-colored dog as it bounds across the open beach." — good fit; matches the sunny, energetic, bounding motion visible.
- sad: "A single light-colored dog moves slowly across the vast, empty expanse of the beach." — directly contradicts the image: the dog is clearly sprinting (tail up, paw raised, mouth open from exertion), not moving slowly. Same failure mode as the snow-dog image (#3): the model downshifts the verb to "slow/calm" to hit "sad" regardless of the actual depicted motion.
- tense: "A light-colored dog sprints along the narrow, wet edge where the ocean hits the sand." — accurate and well-observed; correctly identifies the dog is running near the wet sand/surf line. One of the better-grounded captions across the whole set.
- romantic: "The light-colored dog moves gracefully, its coat glowing against the soft sand of the beach." — contradicts the pose again: this is a fast, almost frantic-looking sprint, not a graceful glide. Nearly identical failure to the "glides through the snow" romantic caption in image #3 — the same false-smoothing move applied to a different photo.
- humorous: "A light-colored dog makes a full-speed attempt to cover the entire length of the beach." — decent, plausible comic exaggeration that's at least consistent with the dog's visible speed.

Emotion-guessability: joyful and tense land well. Sad and romantic both fail by mischaracterizing the physical action as slow/graceful when it's a sprint — this is now a confirmed cross-image pattern (also seen in image #3, the snow dog), not a one-off: whenever the source photo is a high-energy action shot, the sad and romantic registers get produced by falsely inserting "slowly"/"gracefully"/"glides" rather than reframing the same accurate action emotionally.

Person vs. template: "light-colored dog" (lifted verbatim from the source caption) appears in all 5; "beach" in all 5. Skeleton "A light-colored dog [verb clause] ... beach" is essentially identical in shape to image #3's "A brown and white dog [verb clause] ... snow" — suggesting the underlying generation is following one fixed template per source-caption structure regardless of image content.

## 10. 1072153132_53d2bb1b60.jpg
Image: A black-and-white-and-tan collie-type dog leaping in a grassy field, mouth reaching up toward a yellow-green tennis ball just above its nose, mid-jump, all four legs off/leaving the ground. Plain green field, no other objects, bright and clearly a happy, playful moment.

- joyful: "A black and white dog springs into the air to snag a toy." — accurate and fits fine.
- sad: "The black and white dog is alone in the field, catching a toy." — relies purely on the incidental "alone" crutch; the actual scene is an exuberant, playful leap-for-a-ball moment, About as far from a sad mood as this dataset gets, and the caption does nothing to acknowledge that mismatch — it just states "alone" and calls it sad.
- tense: "A black and white dog lunges upward for a toy midair." — accurate, "lunges" captures the effort in the jump reasonably.
- romantic: "A black and white dog leaps gracefully, catching a toy in the air." — reasonably accurate to the pose (a controlled jump can look graceful) but still doesn't read as romantic — just aesthetic description, same recurring issue.
- humorous: "The black and white dog has found a toy while flying through the air." — "flying through the air" is a fun exaggeration that fits; "found" is a slightly odd/illogical word choice (implies stumbling on the toy mid-flight) but reads as intentionally silly enough to work.

Emotion-guessability: joyful, tense, humorous all land reasonably. Sad fails again, and for the same reason as images #3, #4, and #9 — dropping the word "alone" into an otherwise joyful, high-energy scene without adjusting anything else about the sentence. Romantic again fails to actually read as romantic.

Person vs. template: "black and white dog" verbatim in all 5 (matches the source caption's phrase); "toy" and "air/midair" recur across 4/5. Skeleton nearly identical to image #4's dog-catching-toy record — same source-caption shape, same generated template.

CROSS-IMAGE PATTERN CONFIRMED: Checking back across all 10 images, the word "alone" appears verbatim in the sad caption for at least 6 of them (images 2, 3, 4, 6, 8, 10) — and in two of those (image 6's tent-setup, image 8's two-workers-on-a-beam) it directly contradicts the photo, which clearly shows multiple people together. In the other cases (3, 4, 10 — all energetic, happy animal action shots) it's used as a rote sadness trigger with zero regard for whether the rest of the image supports a somber read. "Alone" is functioning as a fill-in-the-blank crutch word for the sad register, not a considered description.

---

# OVERALL VERDICT

**Gut call: not ready to train on as-is. The prompt/pipeline needs fixing first.** The captions are fluent and mostly grammatical, and joyful/tense/humorous are usable more often than not — but sad and romantic are broken often enough, and in ways specific enough (factual contradiction, not just weak tone), that training on this raw file would teach a model to hallucinate isolation and default to purple prose whenever those two emotion tags are requested. It's a good first draft, not a training-ready dataset.

**Single biggest weakness:** the sad register is being generated by inserting the word "alone" (or "empty"/"silence") into the sentence as a mechanical trigger, independent of whether the image supports it — and independent of whether it's even true. Out of the 10 images checked, "alone" appears verbatim in the sad caption for 6 of them:
- Image 103205630 (tent): "A single tent sits alone on the vast ice, waiting to be set up" — while two people are visibly, actively setting the tent up together in the photo. This isn't a tone miss, it's inventing the wrong action to erase the people from a scene that clearly contains them.
- Image 1055753357 (beam workers): "taking a quiet break alone from their work" — while two workers are sitting shoulder-to-shoulder in frame.
- Images 101654506, 1020651753, 1072153132 (all joyful leaping/running dogs): "alone" is bolted onto an otherwise energetic, happy action shot with no other adjustment, so the caption reads as a factual footnote, not a mood.

This tells me the sad-generation step isn't reasoning about the image at all for this register — it's pattern-matching "no people/animals named as companions → say alone," which breaks the moment there's more than one subject in frame, and reads as inert even when there's only one.

**Weakest register: sad**, followed closely by **romantic**. Romantic failed to read as romantic in essentially every single one of the 10 images — it never produces actual romantic content, just a generic soft/gentle/graceful rephrase ("softly ascends," "hands gentle against the ropes," "leans into the cold, silent embrace," "moves gracefully," "leaps gracefully"). Worse, in at least two cases it also hallucinates: image 1048710776 turns five independent people sitting on a wall into "a couple of friends lean into each other," and image 102455176's romantic caption is verbatim the exact phrase the pipeline's own rejection filter already flagged as bad ("figurative/abstract phrasing: the cold, silent embrace") — and it shipped in the final data anyway. That's not just a prompt-quality issue, it's a pipeline-integrity bug: the filter caught it and the bad text still made it downstream.

Sad edges out romantic as "worst" only because its failures are more numerous and more clearly contradict the actual photo (erasing visible people), whereas romantic's failures are more about never landing the register at all.

**Specific prompt changes I'd make:**
1. Ban "alone" as a default sad-mood filler. Require the sad rewrite to point at a specific, visually-grounded detail (posture, lighting, distance, expression) rather than asserting solitude — and never assert solitude when the record shows 2+ people/animals.
2. Give "romantic" an actual definition in the prompt. Right now it's being treated as a synonym for "gentle/soft/graceful," which is why it never differs meaningfully from a watered-down version of the other captions. Either anchor it to something concrete (proximity between subjects, warmth of light, intimacy of framing) or drop it as a register for solo-subject/no-people images where it structurally cannot apply — a solo toddler or a solo dog is not a romantic subject no matter what adjective you use.
3. Fix the rejection filter so a flagged output actually gets regenerated or dropped, not shipped as-is (see image 102455176's romantic caption).
4. Stop verb-swapping past the point of accuracy for sad/romantic: when the source photo is a high-energy action shot (a sprinting dog, a mid-air leap), don't force "moves slowly" or "glides"/"gracefully" — the model should not be allowed to invert the depicted physical motion just to hit a target adjective. Two separate images (101654506, 1057251835) show this exact failure with near-identical language ("glides"/"moves gracefully" over dogs that are visibly sprinting/leaping wildly).
5. Loosen the lexical template. Across almost every record, the anchor noun phrase from the source caption ("black and white dog," "entryway stairs," "beam," "ledge") is repeated verbatim in all 5 rewrites with only a verb or adjective swapped around it. It reads as mad-libs. Instruct the model to vary sentence structure/subject-first vs. scene-first framing across the 5 registers, not just the emotional adjective.
