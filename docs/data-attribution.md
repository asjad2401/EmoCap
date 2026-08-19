# Data provenance and attribution

A factual record of every external dataset and model this project uses: where it came
from, how it was accessed, what its licence permits as we understand it, and what we do
and do not redistribute. Kept current so the paper's data statement can be written from
it directly.

**This is a provenance record, not legal advice.** Two items below are flagged as needing
verification before any public release of derived data — see *Open questions*.

Last updated 2026-08-19.

---

## 1. Flickr8k — primary images and human reference captions

| | |
|---|---|
| **Content used** | 8,091 photographs; 5 human-written captions per image; the official Hodosh et al. train/dev/test splits |
| **Role** | Source images, and the neutral human captions that our generated captions are rewrites of |
| **Access** | Standard research distribution |
| **Citation** | Hodosh, M., Young, P., & Hockenmaier, J. (2013). *Framing image description as a ranking task: Data, models and evaluation metrics.* Journal of Artificial Intelligence Research, 47, 853–899. |

**Redistribution:** we do not redistribute Flickr8k images. They are gitignored
(`/data/flickr8k/`) and never committed. Underlying photographs remain the property of
their Flickr uploaders under their individual terms.

---

## 2. Personality-Captions — human-written style-conditioned captions

| | |
|---|---|
| **Content used** | Human captions conditioned on a personality trait, for the 23 traits that map onto our five registers (~20,515 captions) |
| **Role** | The human comparison arm, and the calibration of the ceiling gate |
| **Access** | `https://parl.ai/downloads/personality_captions/personality_captions.tgz`, downloaded 2026-08-19, sha256 verified against the value published in ParlAI's `parlai/tasks/personality_captions/build.py`: `e0979d3ac0854395ee74f2c61a6bc467838cc292c3a9a62e891d8230d3a01365` |
| **Licence** | ParlAI is MIT-licensed; the data is distributed publicly by its authors without a separate agreement or form |
| **Citation** | Shuster, K., Humeau, S., Hu, H., Bordes, A., & Weston, J. (2019). *Engaging Image Captioning via Personality.* CVPR. |

**Redistribution:** the downloaded archive is gitignored. We commit no captions from it.

---

## 3. YFCC100M / Multimedia Commons — images referenced by Personality-Captions

Personality-Captions ships `image_hash` values only; the images live elsewhere.

| | |
|---|---|
| **Content used** | The subset of images referenced by the captions in §2 (~20,515 images) |
| **Role** | Visual input for CLIP feature extraction on the human arm |
| **Access** | `https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/{h[:3]}/{h[3:6]}/{h}.jpg`, read without credentials. Listed on the AWS Registry of Open Data: https://registry.opendata.aws/multimedia-commons/ |
| **Licence** | Every image in YFCC100M was published on Flickr under a Creative Commons licence. **The specific variant differs per image** (CC-BY, CC-BY-NC, CC-BY-SA, and others), so no single blanket statement covers the set |
| **Citation** | Thomee, B., Shamma, D. A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., & Li, L.-J. (2016). *YFCC100M: The New Data in Multimedia Research.* Communications of the ACM, 59(2), 64–73. |

**What we do:** download images, compute CLIP ViT-B/32 features, and use the features.

**What we do not do:** redistribute, republish, or commit any image, in whole or as a
crop or thumbnail. This is the deliberate position — because per-image licence variants
differ, redistribution would require per-image licence tracking, whereas feature
extraction does not. Downloaded images are gitignored.

**On "permission":** ParlAI's downloader prompts the user to confirm they have obtained
permission per the Multimedia Commons pages. That prompt is a self-attestation, not a
credential check; the S3 bucket is an AWS Open Data public dataset and was read without
authentication. The separate YFCC100M *metadata* file (`yfcc100m_dataset.bz2`),
historically requested through Yahoo Webscope, **is not used by this project** and was
never obtained. If a formal record is wanted later, that registration is the route — but
nothing here depends on it.

---

## 4. Generated captions — our synthetic corpus

| | |
|---|---|
| **Content** | ~202,000 emotion-conditioned captions, each a rewrite of a Flickr8k human caption |
| **Producer** | Google `gemini-3.1-flash-lite` via the Gemini Batch API, prompt version v5 (`prompt_sha256` recorded in every run manifest under `runs/`) |
| **Derived from** | Flickr8k human captions and images (§1) |

**Status in this repository:** the full corpus is gitignored. Small audit samples
(`data/generated/captions_audit_*.jsonl`, 5 × 50 images) are committed as research
evidence, because generation runs at temperature 0.9 and cannot be reproduced
byte-identically.

**Three images excluded.** The generator declined to process three Flickr8k images, all
showing young children in or near water, minimally clothed — recorded in
`data/generated/excluded_images.json`. No attempt was made to circumvent that filtering.
The exclusion is content-correlated rather than random and is disclosed as a limitation.

---

## 5. Archived v1 pilot artifact — Moondream factual captions

Descriptions of all 8,091 Flickr8k images generated by Moondream during the v1 pilot,
retained at `archive/v1-pilot/data/v1_moondream_factual_captions.csv.gz`.

**Role:** used **only** as an offline checker, to detect generated captions that
contradict their own photograph. Deliberately **not** used as model input or as caption
targets — `configs/prereg.lock.yaml` records `neutral_source: human_annotations  # NOT a
VLM` precisely to keep a VLM's output out of the dataset.

---

## 6. Pretrained models

| model | role | licence |
|---|---|---|
| OpenAI CLIP ViT-B/32 | frozen image encoder | MIT |
| GPT-2 | decoder (Track B) | MIT |
| DistilRoBERTa | held-out register classifier | Apache 2.0 |

None are redistributed; they are fetched from their upstream sources at run time.

---

## 7. Not used

**ArtEmis** (Achlioptas et al., CVPR 2021) was evaluated as a candidate for the human
comparison and **not used**; Personality-Captions was chosen instead as a closer match to
the task. No ArtEmis data was requested or downloaded, and its access form was not
submitted.

**FlickrStyle10K** (Gan et al., CVPR 2017) was considered and not obtained. Only its 7K
train split is publicly released and no stable distribution URL was found; unvetted
third-party reuploads were deliberately not used.

---

## Open questions — resolve before releasing derived data

1. **Publishing the generated corpus.** The captions were produced through the Gemini
   API. Whether the applicable API terms permit publishing a derived dataset of that
   size, and under what conditions, has **not** been verified. Check the current terms
   before releasing the corpus publicly. Publishing measurements, aggregate statistics
   and the code is a separate and much narrower question than publishing the captions
   themselves.

2. **Per-image licences on the YFCC subset.** Not an issue for the present design, since
   only features are computed and no image is redistributed. It would become one if
   images, crops, or thumbnails ever appeared in a paper figure or a released artifact —
   that would need per-image licence and attribution tracking.
