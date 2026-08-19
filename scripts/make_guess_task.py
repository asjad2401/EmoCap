#!/usr/bin/env python
"""Blind register-guessing task: can a human recover the label the classifier is scored on?

The primary metric is a classifier reading a caption and naming which of five registers was
requested. It scores 0.775 on the v5 corpus. **Nobody has ever measured what a human scores
on the same task with the same data** -- so there is no way to tell whether 0.775 is close to
a ceiling or nowhere near it.

    uv run python scripts/make_guess_task.py --mode text  --n 100
    uv run python scripts/make_guess_task.py --mode image --n 60

Two modes, and the contrast between them is the interesting part:

* ``text``  -- caption alone. This is EXACTLY the classifier's task, so the score is directly
  comparable to its accuracy.
* ``image`` -- caption plus the photograph. If seeing the image RAISES the score, part of the
  register lives in the caption-image relationship rather than in the text, and a text-only
  classifier is structurally unable to see it. If it makes no difference, the register is
  purely textual and the classifier is not missing anything a reader has.

Design points that matter:

* Registers are **balanced** (n/5 each) and the order is **shuffled**, so neither frequency
  nor position leaks the answer.
* **No feedback** is given. Telling the guesser whether they were right would train them
  mid-task and bias later answers.
* Nothing in the page names the intended register -- the label is only in the exported
  results, and the answer key is written to a separate file this page never loads.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.runtime import load_config  # noqa: E402

PAGE = """<!doctype html>
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
main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:28px 18px;max-width:760px;margin:0 auto;width:100%}
#img{max-width:100%;max-height:42vh;border-radius:10px;border:1px solid var(--line);
  margin-bottom:22px;display:none}
#cap{font-size:23px;line-height:1.45;text-align:center;margin:0 0 34px;font-weight:450}
.opts{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.opts button{font:inherit;font-size:15px;padding:13px 22px;border:1px solid var(--line);
  border-radius:10px;background:var(--card);color:var(--fg);cursor:pointer;min-width:118px}
.opts button:hover{border-color:var(--acc);background:var(--acc);color:#fff}
.opts button b{display:block;font-size:11px;color:var(--mut);font-weight:400;margin-top:3px}
.opts button:hover b{color:#fff;opacity:.8}
footer{border-top:1px solid var(--line);padding:12px 18px;display:flex;gap:14px;
  align-items:center;justify-content:center}
button.sec{font:inherit;padding:7px 15px;border:1px solid var(--line);border-radius:8px;
  background:var(--card);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}
.intro{max-width:620px}
.intro h2{font-size:19px;margin:0 0 12px}
.intro p{color:var(--mut);font-size:15px}
.done{text-align:center}.done h2{font-size:22px}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;
  padding:1px 6px;background:var(--card)}
</style></head><body>
<header><h1>__TITLE__</h1><span class="pill" id="pos"></span>
<div class="bar"><i id="fill"></i></div></header>
<main id="main"></main>
<footer>
  <button class="sec" id="back">&larr; back</button>
  <button class="sec" id="skip">skip</button>
  <button class="pri" id="exp">Export results</button>
</footer>
<script>
const D = __DATA__;
const KEY = 'emocap-guess-' + D.mode + '-' + D.seed;
let S = JSON.parse(localStorage.getItem(KEY) || '{"answers":{},"i":0}');
const save = () => localStorage.setItem(KEY, JSON.stringify(S));
const M = document.getElementById('main');
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const HINT = {joyful:'1', sad:'2', tense:'3', romantic:'4', humorous:'5'};

function intro(){
  M.innerHTML = `<div class="intro">
    <h2>${D.mode === 'image' ? 'Caption + photograph' : 'Caption only'} &mdash; guess the register</h2>
    <p>You will see ${D.items.length} captions${D.mode === 'image' ? ', each with its photograph' : ''}.
    Each was written to convey exactly one of five registers. Pick which one you think was asked for.</p>
    <p><b>Go on instinct.</b> First reaction is the measurement &mdash; deliberating changes what
    is being tested. There is no feedback, deliberately: knowing whether you were right would
    train you mid-task.</p>
    <p>Keys <kbd>1</kbd>&ndash;<kbd>5</kbd> work. Roughly ${Math.ceil(D.items.length/10)} minutes.</p>
    <p style="margin-top:20px"><button class="sec pri" onclick="go()">Start</button></p></div>`;
}
function go(){ render(); }

function render(){
  if (S.i >= D.items.length) return finish();
  const it = D.items[S.i];
  document.getElementById('pos').textContent = `${S.i+1} / ${D.items.length}`;
  document.getElementById('fill').style.width = (100*S.i/D.items.length)+'%';
  M.innerHTML =
    (D.mode === 'image' ? `<img id="img" src="${esc(it.rel)}" style="display:block" alt="">` : '')
    + `<p id="cap">&ldquo;${esc(it.text)}&rdquo;</p><div class="opts">`
    + D.registers.map(r => `<button data-r="${r}">${r}<b>${HINT[r]}</b></button>`).join('')
    + `</div>`;
  M.querySelectorAll('[data-r]').forEach(b => b.onclick = () => answer(b.dataset.r));
}
function answer(r){
  S.answers[D.items[S.i].id] = r; S.i++; save(); render();
}
function finish(){
  document.getElementById('fill').style.width = '100%';
  const n = Object.keys(S.answers).length;
  M.innerHTML = `<div class="done"><h2>Done &mdash; ${n} answered</h2>
    <p style="color:var(--mut)">Press <b>Export results</b> and hand the file back.
    Your score is not shown here: computing it needs the answer key, which this page
    deliberately does not contain.</p></div>`;
}
document.getElementById('back').onclick = () => { if (S.i>0){ S.i--; save(); render(); } };
document.getElementById('skip').onclick = () => { S.i++; save(); render(); };
document.addEventListener('keydown', e => {
  const k = ['1','2','3','4','5'].indexOf(e.key);
  if (k >= 0 && S.i < D.items.length) answer(D.registers[k]);
  if (e.key === 'ArrowLeft') document.getElementById('back').click();
});
document.getElementById('exp').onclick = () => {
  const out = {tool:'make_guess_task.py', mode:D.mode, seed:D.seed, source:D.source,
    n_items:D.items.length, answered:Object.keys(S.answers).length,
    exported_at:new Date().toISOString(), answers:S.answers};
  const b = new Blob([JSON.stringify(out,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = `emocap-guess-${D.mode}-${D.seed}.json`; a.click();
};
if (Object.keys(S.answers).length || S.i) render(); else intro();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "image"], required=True)
    ap.add_argument("--n", type=int, default=100, help="items; rounded down to a multiple of 5")
    ap.add_argument("--store", default=None, help="caption store (default: the v5 corpus)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--exclude-images", default=None,
                    help="JSON list of image_ids to keep OUT of the task. Use this whenever "
                         "the guesser has already seen those images WITH their labels in a "
                         "quality audit -- otherwise the task measures recall of the audit "
                         "rather than legibility of the caption.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--key", default=None,
                    help="where the answer key goes; default runs/guess/{mode}_key.json. "
                         "Set it when building a SECOND task in the same mode, or the first "
                         "task's key is overwritten and its export becomes unscoreable.")
    args = ap.parse_args()

    cfg = load_config("data")
    store = Path(args.store) if args.store else ROOT / cfg["paths"]["raw_generations"]
    if not store.is_absolute():
        store = ROOT / store
    recs = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]

    blocked: set[str] = set()
    if args.exclude_images:
        bp = Path(args.exclude_images)
        if not bp.is_absolute():
            bp = ROOT / bp
        blocked = set(json.loads(bp.read_text()))
        before = len({r["image_id"] for r in recs})
        recs = [r for r in recs if r["image_id"] not in blocked]
        after = len({r["image_id"] for r in recs})
        print(f"excluded {before - after} already-seen images -> {after} available")

    seed = args.seed if args.seed is not None else (20260820 if args.mode == "text" else 20260821)
    per = args.n // len(EMOTIONS)
    rng = random.Random(seed)

    # Balanced by register, and each caption from a DIFFERENT image so no image contributes
    # two items -- otherwise a guesser who recognises the scene gets a free comparison.
    pool: dict[str, list[tuple[str, str]]] = {e: [] for e in EMOTIONS}
    for r in recs:
        for e in EMOTIONS:
            t = str(r["captions"].get(e, "")).strip()
            if t:
                pool[e].append((r["image_id"], t))
    used: set[str] = set()
    items = []
    for e in EMOTIONS:
        cand = pool[e][:]
        rng.shuffle(cand)
        taken = 0
        for img, text in cand:
            if img in used:
                continue
            used.add(img)
            items.append({"id": f"{e[0]}{taken}_{img[:10]}", "text": text,
                          "truth": e, "image_id": img,
                          "rel": str(Path(cfg["paths"]["images_dir"]) / img)})
            taken += 1
            if taken >= per:
                break
        if taken < per:
            raise SystemExit(
                f"only {taken} unique-image captions available for '{e}', need {per}. "
                f"The one-image-per-item rule is not negotiable -- an image seen twice hands "
                f"the guesser a free comparison. Lower --n or generate more images."
            )
    rng.shuffle(items)

    # The answer key goes to a SEPARATE file the page never loads.
    key = {it["id"]: {"truth": it["truth"], "image_id": it["image_id"], "text": it["text"]}
           for it in items}
    public = [{k: v for k, v in it.items() if k != "truth"} for it in items]

    data = {"mode": args.mode, "seed": seed, "registers": list(EMOTIONS),
            "source": store.name, "items": public}
    title = ("Guess the register — caption only" if args.mode == "text"
             else "Guess the register — caption + photograph")
    out = ROOT / (args.out or f"GUESS-{args.mode.upper()}.html")
    out.write_text(PAGE.replace("__DATA__", json.dumps(data)).replace("__TITLE__", title),
                   encoding="utf-8")
    kp = ROOT / (args.key or f"runs/guess/{args.mode}_key.json")
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps({"mode": args.mode, "seed": seed, "source": store.name,
                              "excluded_images": sorted(blocked), "key": key}, indent=2))

    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  {len(items)} items, {per} per register, from {store.name}")
    print(f"  answer key -> {kp}  (the page never loads it)")
    print(f"\n  open {out}")


if __name__ == "__main__":
    main()
