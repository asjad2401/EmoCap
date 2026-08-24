#!/usr/bin/env python
"""Build the registered human evaluation. 150 items, two 5-point scales, blinded.

    uv run --extra vision python scripts/make_human_eval.py
    uv run --extra vision python scripts/make_human_eval.py --n 150 --checks 12

Writes a single self-contained HTML file a rater opens in any browser, and a separate
answer key this project keeps. The page never loads the key.

`configs/prereg.lock.yaml` registers this as CONFIRMATORY: 150 items, 3 raters, tone-match
and grounding on 1-5 scales, Krippendorff's alpha, blinded, with attention checks, and
**run once on final models only**. Running it twice is a deviation, so everything the
selection depends on is fixed here rather than decided per invocation.

What a rater sees, and what they never see
------------------------------------------
A photograph, one caption, and the register that caption was *asked* to convey. They give
two independent 1-5 scores: does the caption carry that tone, and does it describe that
photograph. The requested register has to be shown -- tone-match is meaningless without it.

They never see which arm produced the caption, and never see whether an item is an
attention check. That is the blinding that matters here: the study compares arms, so the
rater must not be able to tell them apart or to tell which items are being used to check
them.

No photograph appears twice in the whole task
---------------------------------------------
The three arms caption the SAME 8,047 Flickr8k images. Sampling each arm independently
would show one scene under two systems, side by side, which hands the rater a comparison
and destroys the blind. Images are therefore partitioned: each photograph belongs to
exactly one arm's pool, dealt round-robin from a hash ordering. The attention checks draw
from a further reserved slice, so their photographs are unused too.

Which arms, and why not all six
-------------------------------
S_paired25, S_paired5 and V1_paired5 -- the three arms that score above chance. The three
1-caption-per-image arms sit at their own keyword anchor, meaning they write essentially
register-free text; spending half the rating budget on captions nobody disputes buys
nothing. The exclusion is a choice made before any rating exists and must be reported.

The attention checks, and their threshold, are fixed BEFORE any rating exists
----------------------------------------------------------------------------
The lock says `attention_checks: true` and nothing more, so the design is set here and
recorded in `docs/deviations.md` rather than invented after seeing rater behaviour.

* **wrong-register** -- a real caption shown under a register it was not written for, and
  only ever a distant one (joyful shown as sad, tense shown as romantic). An attentive
  rater scores TONE low.
* **wrong-image** -- a real caption shown against an unrelated photograph. An attentive
  rater scores GROUNDING low.

A check passes when the relevant scale is <= 2. A rater who fails more than half of their
checks is FLAGGED, not silently dropped -- excluding a rater is a judgement to be made and
reported, not applied by a script. Checks are additional to the registered 150, so exactly
150 scored items reach the analysis.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.runtime import load_config  # noqa: E402

SALT = "humaneval-v1"

#: Arms whose captions are rated. Not the registered six -- see the module docstring.
ARMS = ("S_paired25", "S_paired5", "V1_paired5")

#: Registers far enough apart that a mismatch is unambiguous to an attentive reader.
#: A joyful caption shown as "humorous" would be a fair disagreement, not a check.
DISTANT = {"joyful": "sad", "sad": "joyful", "tense": "romantic",
           "romantic": "tense", "humorous": "sad"}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e0ddd8;--card:#fff;--acc:#3b5bdb}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e6e3;--mut:#9a9a9a;
  --line:#2e3033;--card:#1f2124;--acc:#7b93f0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  display:flex;flex-direction:column;min-height:100vh}
header{border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:14px;
  align-items:center}
h1{font-size:14px;margin:0;font-weight:650}
.bar{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);width:0%;transition:width .15s}
.pill{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
main{flex:1;display:flex;flex-direction:column;align-items:center;
  padding:24px 18px;max-width:720px;margin:0 auto;width:100%}
img#photo{max-width:100%;max-height:38vh;border-radius:10px;border:1px solid var(--line);
  margin-bottom:18px}
.cap{font-size:20px;line-height:1.45;text-align:center;margin:0 0 6px;font-weight:450}
.asked{font-size:13px;color:var(--mut);text-align:center;margin:0 0 26px}
.asked b{color:var(--acc);font-weight:650;text-transform:lowercase}
.q{width:100%;margin-bottom:22px}
.q p{margin:0 0 8px;font-size:14px}
.q .sub{color:var(--mut);font-size:12.5px;margin:-4px 0 9px}
.scale{display:flex;gap:8px}
.scale button{flex:1;font:inherit;font-size:15px;padding:11px 0;border:1px solid var(--line);
  border-radius:9px;background:var(--card);color:var(--fg);cursor:pointer}
.scale button:hover{border-color:var(--acc)}
.scale button.on{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:650}
.ends{display:flex;justify-content:space-between;font-size:11.5px;color:var(--mut);
  margin-top:5px}
footer{border-top:1px solid var(--line);padding:12px 18px;display:flex;gap:12px;
  align-items:center;justify-content:center;flex-wrap:wrap}
button.sec{font:inherit;padding:8px 16px;border:1px solid var(--line);border-radius:8px;
  background:var(--card);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}
button:disabled{opacity:.4;cursor:not-allowed}
.intro{max-width:620px}.intro h2{font-size:20px;margin:0 0 12px}
.intro p{color:var(--mut);font-size:15px}
.intro ul{color:var(--mut);font-size:15px;padding-left:20px}
.intro label{display:block;margin:18px 0 6px;font-size:14px;color:var(--fg)}
.intro input{font:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:8px;
  background:var(--card);color:var(--fg);width:260px}
.done{text-align:center;max-width:600px}.done h2{font-size:22px}
textarea{width:100%;height:170px;font:12px ui-monospace,monospace;margin-top:12px;
  border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);
  padding:10px}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;
  padding:1px 6px;background:var(--card)}
</style></head><body>
<header><h1>__TITLE__</h1><span class="pill" id="pos"></span>
<div class="bar"><i id="fill"></i></div></header>
<main id="main"></main>
<footer>
  <button class="sec" id="back">&larr; back</button>
  <button class="sec" id="next">next &rarr;</button>
  <button class="pri" id="exp">Finish &amp; export</button>
</footer>
<script>
const D = __DATA__;
const KEY = 'emocap-humaneval-' + D.build + '-v1';
let S = JSON.parse(localStorage.getItem(KEY) || '{"a":{},"i":0,"rater":""}');
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch(e){} };
const M = document.getElementById('main');
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SCALES = [
  {k:'tone', q:'How well does the caption convey the requested tone?',
   sub:'Judge the writing only. A caption can convey the tone perfectly and still describe the photo badly.',
   lo:'1 — not at all', hi:'5 — completely'},
  {k:'ground', q:'How well does the caption describe this photograph?',
   sub:'Judge accuracy against the image only. Ignore the tone entirely here.',
   lo:'1 — unrelated', hi:'5 — accurate'}
];

function intro(){
  M.innerHTML = `<div class="intro">
    <h2>Rating captions</h2>
    <p>You will see <b>${D.items.length}</b> photographs. Each comes with one caption and the
    <b>tone that caption was asked to convey</b>. You give two separate scores, 1 to 5.</p>
    <ul>
      <li><b>Tone</b> — does the caption actually carry that tone?</li>
      <li><b>Description</b> — does the caption accurately describe the photo?</li>
    </ul>
    <p>Keep the two judgements apart. A caption can nail the tone while describing the wrong
    scene, or describe the scene perfectly in flat, toneless language. That difference is
    exactly what this is measuring.</p>
    <p>Go with your first reaction. There is no right answer to look for and no feedback.
    You can stop and come back &mdash; your progress is saved in this browser.</p>
    <p>Keys <kbd>1</kbd>&ndash;<kbd>5</kbd> score the highlighted question,
    <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move between items.</p>
    <label for="rid">Your name or initials</label>
    <input id="rid" placeholder="e.g. AA" value="${esc(S.rater||'')}">
    <p style="margin-top:22px"><button class="pri sec" id="go">Start</button></p>
  </div>`;
  const go = () => {
    const v = document.getElementById('rid').value.trim();
    if(!v){ document.getElementById('rid').focus(); return; }
    S.rater = v; save(); render();
  };
  document.getElementById('go').onclick = go;
  document.getElementById('rid').onkeydown = e => { if(e.key==='Enter') go(); };
}

function render(){
  if(!S.rater){ intro(); return; }
  if(S.i >= D.items.length){ done(); return; }
  const it = D.items[S.i];
  const a = S.a[it.id] || {};
  document.getElementById('pos').textContent = `${S.i+1} / ${D.items.length}`;
  document.getElementById('fill').style.width =
    (100*Object.keys(S.a).filter(k=>S.a[k].tone&&S.a[k].ground).length/D.items.length)+'%';
  M.innerHTML = `
    <img id="photo" src="${it.img}" alt="">
    <p class="cap">${esc(it.text)}</p>
    <p class="asked">requested tone: <b>${esc(it.asked)}</b></p>
    ${SCALES.map(s => `<div class="q" data-k="${s.k}">
      <p>${s.q}</p><p class="sub">${s.sub}</p>
      <div class="scale">${[1,2,3,4,5].map(n =>
        `<button data-n="${n}" class="${a[s.k]===n?'on':''}">${n}</button>`).join('')}</div>
      <div class="ends"><span>${s.lo}</span><span>${s.hi}</span></div></div>`).join('')}`;
  M.querySelectorAll('.q').forEach(q => {
    q.querySelectorAll('button').forEach(b => {
      b.onclick = () => {
        const rec = S.a[it.id] || (S.a[it.id] = {});
        rec[q.dataset.k] = +b.dataset.n;
        rec.t = Date.now();
        save();
        // Advance only once BOTH scales are answered, so the second question is never
        // skipped past by a fast click on the first.
        if(rec.tone && rec.ground){ S.i++; save(); render(); } else { render(); }
      };
    });
  });
}

function done(){
  const payload = JSON.stringify({build: D.build, rater: S.rater,
    answers: S.a, finished: new Date().toISOString()}, null, 1);
  const n = Object.keys(S.a).filter(k=>S.a[k].tone&&S.a[k].ground).length;
  M.innerHTML = `<div class="done"><h2>Done &mdash; thank you</h2>
    <p>${n} of ${D.items.length} items rated.</p>
    <p>Click <b>Download</b> and send the file back. If the download is blocked, copy the
    text below into an email instead.</p>
    <p><button class="pri sec" id="dl">Download results</button>
       <button class="sec" id="cp">Copy to clipboard</button></p>
    <textarea readonly id="ta">${esc(payload)}</textarea></div>`;
  document.getElementById('dl').onclick = () => {
    const b = new Blob([payload], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `humaneval-${D.build}-${S.rater.replace(/[^A-Za-z0-9]/g,'')}.json`;
    document.body.appendChild(a); a.click(); a.remove();
  };
  document.getElementById('cp').onclick = () => {
    const ta = document.getElementById('ta'); ta.select();
    navigator.clipboard.writeText(payload).catch(()=>document.execCommand('copy'));
  };
}

document.getElementById('back').onclick = () => { if(S.i>0){S.i--;save();render();} };
document.getElementById('next').onclick = () => {
  if(S.i < D.items.length){S.i++;save();render();} };
document.getElementById('exp').onclick = () => { S.i = D.items.length; save(); done(); };
document.addEventListener('keydown', e => {
  if(!S.rater || S.i >= D.items.length) return;
  if(e.key === 'ArrowLeft'){ if(S.i>0){S.i--;save();render();} return; }
  if(e.key === 'ArrowRight'){ S.i++; save(); render(); return; }
  if(!'12345'.includes(e.key)) return;
  const it = D.items[S.i], rec = S.a[it.id] || (S.a[it.id] = {});
  // Number keys fill tone first, then grounding -- the same order they are shown in.
  const k = rec.tone ? 'ground' : 'tone';
  rec[k] = +e.key; rec.t = Date.now(); save();
  if(rec.tone && rec.ground){ S.i++; save(); }
  render();
});
render();
</script></body></html>
"""


def _h(*parts: str) -> int:
    return int(hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:12], 16)


def item_id(arm: str, image_id: str, emotion: str, kind: str) -> str:
    """Opaque id. It travels to the browser, so it must not encode the arm, the register,
    or whether the item is a check -- all three are exactly what the rater is blind to."""
    return hashlib.sha256(
        f"{SALT}|{arm}|{image_id}|{emotion}|{kind}".encode()).hexdigest()[:12]


def load_arm_predictions(arm: str, runs_dir: Path) -> dict[str, dict[str, str]]:
    """``image_id -> {register: generated}`` over every fold's held-out predictions.

    S_paired25 holds five source captions per (image, register); the first is kept, chosen
    by fold order rather than at random, because the arm is being sampled for a rating task
    and not measured here.
    """
    out: dict[str, dict[str, str]] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{arm}-f") or d.name.endswith("-nc"):
            continue
        path = d / "predictions.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} is missing -- it is gitignored; pull the run first")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(r["image_id"], {}).setdefault(r["emotion"], r["generated"])
    return out


def encode_image(path: Path, max_dim: int, quality: int) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="scored items; defaults to the registered human_eval.items")
    ap.add_argument("--checks", type=int, default=12,
                    help="attention checks, ADDITIONAL to --n")
    ap.add_argument("--runs", default="runs/arms")
    ap.add_argument("--out", default="HUMAN-EVAL.html")
    ap.add_argument("--key", default="runs/human-eval/key.json")
    ap.add_argument("--max-dim", type=int, default=560)
    ap.add_argument("--quality", type=int, default=72)
    args = ap.parse_args()

    lock = load_config("prereg.lock")
    spec = lock["human_eval"]
    n = args.n or spec["items"]
    if n % (len(ARMS) * len(EMOTIONS)):
        raise SystemExit(f"--n {n} does not divide evenly into {len(ARMS)} arms x "
                         f"{len(EMOTIONS)} registers")
    per_arm = n // len(ARMS)
    per_cell = per_arm // len(EMOTIONS)
    if args.checks % 2:
        raise SystemExit("--checks must be even (half wrong-register, half wrong-image)")

    cfg = load_config("data")
    images_dir = ROOT / cfg["paths"]["images_dir"]
    runs_dir = ROOT / args.runs

    preds = {a: load_arm_predictions(a, runs_dir) for a in ARMS}
    shared = sorted(set.intersection(*(set(p) for p in preds.values())))
    shared = [i for i in shared if (images_dir / i).exists()]
    ranked = sorted(shared, key=lambda i: _h(SALT, i))
    print(f"{len(ranked):,} photographs captioned by all {len(ARMS)} arms, images on disk")

    # Deal photographs out round-robin so no image is ever shown under two arms.
    owned: dict[str, list[str]] = {a: [] for a in ARMS}
    for k, img in enumerate(ranked):
        owned[ARMS[k % len(ARMS)]].append(img)
    # A reserved tail, never used by a scored item, supplies the checks.
    reserve = [owned[a].pop() for a in ARMS for _ in range(args.checks)]

    items: list[dict] = []
    for arm in ARMS:
        pool = owned[arm]
        used: set[str] = set()
        for emo in EMOTIONS:
            taken = 0
            for img in sorted(pool, key=lambda i: _h(SALT, arm, emo, i)):
                if img in used:
                    continue
                text = (preds[arm].get(img, {}).get(emo) or "").strip()
                if not text:
                    continue
                used.add(img)
                items.append({"id": item_id(arm, img, emo, "real"), "text": text,
                              "asked": emo, "image_id": img, "arm": arm,
                              "kind": "real", "expect_low": None})
                taken += 1
                if taken >= per_cell:
                    break
            if taken < per_cell:
                raise SystemExit(f"{arm}/{emo}: only {taken} usable items, need {per_cell}")

    # ── attention checks ────────────────────────────────────────────────────
    half = args.checks // 2
    ck = sorted(reserve, key=lambda i: _h(SALT, "check", i))
    for j in range(half):
        arm = ARMS[j % len(ARMS)]
        img = ck[j]
        true_emo = EMOTIONS[j % len(EMOTIONS)]
        text = (preds[arm].get(img, {}).get(true_emo) or "").strip()
        if not text:
            continue
        items.append({"id": item_id(arm, img, true_emo, "wrong_register"), "text": text,
                      "asked": DISTANT[true_emo], "image_id": img, "arm": arm,
                      "kind": "wrong_register", "expect_low": "tone",
                      "true_register": true_emo})
    for j in range(half):
        arm = ARMS[j % len(ARMS)]
        src, decoy = ck[half + j], ck[half + j + half]
        true_emo = EMOTIONS[j % len(EMOTIONS)]
        text = (preds[arm].get(src, {}).get(true_emo) or "").strip()
        if not text:
            continue
        items.append({"id": item_id(arm, src, true_emo, "wrong_image"), "text": text,
                      "asked": true_emo, "image_id": decoy, "arm": arm,
                      "kind": "wrong_image", "expect_low": "ground",
                      "caption_belongs_to": src})

    # One fixed presentation order, hashed rather than shuffled at runtime, so every rater
    # sees the same sequence and a re-build of this file is byte-comparable.
    items.sort(key=lambda it: _h(SALT, "order", it["id"]))

    build = hashlib.sha256(
        "".join(it["id"] for it in items).encode()).hexdigest()[:10]

    print(f"embedding {len({it['image_id'] for it in items}):,} photographs "
          f"at <={args.max_dim}px q{args.quality} ...")
    cache: dict[str, str] = {}
    public = []
    for it in items:
        img = it["image_id"]
        if img not in cache:
            cache[img] = encode_image(images_dir / img, args.max_dim, args.quality)
        public.append({"id": it["id"], "text": it["text"], "asked": it["asked"],
                       "img": cache[img]})

    out = ROOT / args.out
    out.write_text(
        PAGE.replace("__DATA__", json.dumps({"build": build, "items": public}))
            .replace("__TITLE__", "Caption rating"), encoding="utf-8")

    kp = ROOT / args.key
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps({
        "build": build, "salt": SALT, "arms": list(ARMS),
        "registered": {k: spec[k] for k in sorted(spec)},
        "n_scored": sum(1 for it in items if it["kind"] == "real"),
        "n_checks": sum(1 for it in items if it["kind"] != "real"),
        "check_pass_rule": "the expect_low scale is <= 2",
        "rater_flag_rule": "a rater failing more than half of their checks is FLAGGED; "
                           "excluding one is a reported judgement, never automatic",
        "excluded_arms": "the three 1-caption-per-image arms; they sit at their own "
                         "keyword anchor and write essentially register-free text",
        "key": {it["id"]: {k: v for k, v in it.items() if k != "id"} for it in items},
    }, indent=2))

    real = [it for it in items if it["kind"] == "real"]
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(real)} scored items  {dict(Counter(i['arm'] for i in real))}")
    print(f"  registers {dict(Counter(i['asked'] for i in real))}")
    print(f"  {len(items) - len(real)} attention checks "
          f"{dict(Counter(i['kind'] for i in items if i['kind'] != 'real'))}")
    print(f"  images reused: {len(items) - len(cache)} (0 means every photo appears once)")
    print(f"  answer key -> {kp.relative_to(ROOT)}  (the page never loads it)")
    print(f"\n  send {out.name} to {spec['raters']} raters; collect their JSON exports into "
          f"runs/human-eval/")


if __name__ == "__main__":
    main()
