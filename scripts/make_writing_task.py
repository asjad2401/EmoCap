#!/usr/bin/env python
"""Build the off-distribution writing task: humans caption photographs in a given register.

    uv run --extra vision python scripts/make_writing_task.py
    uv run --extra vision python scripts/make_writing_task.py --n 250 --chunks 5

Writes ``WRITING-<k>-of-<n>.html`` plus ``runs/offdist/key.json``.

`metrics.classifier.robustness_reported` registers **250 hand-labelled captions not written
by Gemini**, and §4.1 calls it "the instrument's real measurement error".

**The criticism this answers.** The frozen classifier learned register from 13,170 captions,
8,780 of which Gemini wrote. So a reviewer can say: *it did not learn emotional register, it
learned what Gemini's joyful captions look like*. That is not a nitpick, because the arms
were trained on Gemini text and therefore produce Gemini-flavoured text -- the classifier
could be recognising a house style, and every number in the study would look identical either
way. The keyword anchor cannot catch it either, since it is fitted on the same text.

Only captions from outside that distribution can settle it. Humans writing to the same five
register definitions is the strongest available source: genuinely off-distribution, and
directly parallel to what the models do, because the writer sees the photograph and nothing
else -- no source caption, and never the model's own attempt, which would anchor them.

**Every image is one nobody involved has seen labelled.** The human-evaluation and legibility
tasks' images are excluded, so a writer cannot be reproducing a register they were shown.

**Split into chunks so several people can write in parallel**, and because 250 embedded
photographs in one file is unwieldy. Each chunk is a standalone HTML file with its own slice;
the key covers all of them and the scorer pools whatever comes back.

**What is NOT collected.** No name, no timing beyond a per-item timestamp for sanity checks,
and no opinion about the caption. The writer supplies text; the register is what they were
asked for, which is what makes the label trustworthy without a second annotator.
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

from emocap.data.prompt import EMOTIONS, REGISTERS  # noqa: E402
from emocap.runtime import load_config, rel  # noqa: E402

SALT = "offdist-v1"

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
  padding:22px 18px;max-width:760px;margin:0 auto;width:100%}
img#photo{max-width:100%;max-height:40vh;border-radius:10px;border:1px solid var(--line);
  margin-bottom:16px}
.ask{font-size:15px;text-align:center;margin:0 0 4px}
.ask b{color:var(--acc);font-weight:700}
.guide{font-size:13.5px;color:var(--mut);text-align:center;margin:0 0 16px;
  max-width:620px;line-height:1.5}
textarea#cap{width:100%;min-height:92px;font:inherit;font-size:17px;padding:12px 14px;
  border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);
  resize:vertical}
textarea#cap:focus{outline:none;border-color:var(--acc)}
.count{font-size:12px;color:var(--mut);margin-top:6px;align-self:flex-end}
footer{border-top:1px solid var(--line);padding:12px 18px;display:flex;gap:12px;
  align-items:center;justify-content:center;flex-wrap:wrap}
button{font:inherit;padding:9px 18px;border:1px solid var(--line);border-radius:8px;
  background:var(--card);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}
button:disabled{opacity:.4;cursor:not-allowed}
.intro{max-width:640px}.intro h2{font-size:20px;margin:0 0 12px}
.intro p,.intro li{color:var(--mut);font-size:15px}
.done{text-align:center;max-width:620px}
textarea.out{width:100%;height:170px;font:12px ui-monospace,monospace;margin-top:12px;
  border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);
  padding:10px}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;
  padding:1px 6px;background:var(--card)}
</style></head><body>
<header><h1>__TITLE__</h1><span class="pill" id="pos"></span>
<div class="bar"><i id="fill"></i></div></header>
<main id="main"></main>
<footer>
  <button id="back">&larr; back</button>
  <button id="skip">skip</button>
  <button class="pri" id="next">next &rarr;</button>
  <button id="exp">Finish &amp; export</button>
</footer>
<script>
const D = __DATA__;
const KEY = 'emocap-writing-' + D.chunk + '-' + D.build + '-v1';
let S = JSON.parse(localStorage.getItem(KEY) || '{"a":{},"i":0}');
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch(e){} };
const M = document.getElementById('main');
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function intro(){
  M.innerHTML = `<div class="intro">
    <h2>Write a caption for each photo</h2>
    <p>You will see <b>${D.items.length}</b> photographs. Each one comes with a
    <b>tone</b>. Write a single sentence describing the photo <i>in that tone</i>.</p>
    <ul>
      <li><b>Describe what is actually there.</b> Never invent people, weather, danger or
      relationships that are not in the frame.</li>
      <li><b>One sentence</b> is plenty. Aim for roughly 10&ndash;25 words.</li>
      <li>A short guide to the tone appears under each photo. Follow it loosely &mdash;
      write how you would naturally write.</li>
    </ul>
    <p>There is no right answer and nothing is being judged. Your sentences become test
    material for a classifier, and what matters is only that they are yours.</p>
    <p>Progress saves in this browser, so you can stop and come back.
    <kbd>Ctrl</kbd>+<kbd>Enter</kbd> moves to the next photo.</p>
    <p style="margin-top:20px"><button class="pri" id="go">Start</button></p></div>`;
  document.getElementById('go').onclick = () => { S.started = 1; save(); render(); };
}

function render(){
  if(!S.started){ intro(); return; }
  if(S.i >= D.items.length){ done(); return; }
  const it = D.items[S.i];
  const cur = (S.a[it.id] || {}).text || '';
  const written = Object.values(S.a).filter(v => (v.text||'').trim().length > 0).length;
  document.getElementById('pos').textContent = `${S.i+1} / ${D.items.length}`;
  document.getElementById('fill').style.width = (100*written/D.items.length)+'%';
  M.innerHTML = `
    <img id="photo" src="${it.img}" alt="">
    <p class="ask">Describe this photo in a <b>${esc(it.register)}</b> tone.</p>
    <p class="guide">${esc(D.guide[it.register])}</p>
    <textarea id="cap" placeholder="One sentence describing what you see..."></textarea>
    <div class="count" id="wc"></div>`;
  const ta = document.getElementById('cap');
  ta.value = cur;
  const wc = () => {
    const n = ta.value.trim().split(/\s+/).filter(Boolean).length;
    document.getElementById('wc').textContent = n ? `${n} words` : '';
  };
  wc(); ta.focus();
  ta.oninput = () => {
    const rec = S.a[it.id] || (S.a[it.id] = {});
    rec.text = ta.value; rec.t = Date.now(); save(); wc();
  };
  ta.onkeydown = e => {
    if(e.key === 'Enter' && (e.ctrlKey || e.metaKey)){ e.preventDefault(); adv(1); }
  };
}

function adv(d){ S.i = Math.max(0, S.i + d); save(); render(); }

function done(){
  const items = D.items
    .filter(it => ((S.a[it.id]||{}).text||'').trim())
    .map(it => ({id: it.id, text: S.a[it.id].text.trim(), t: S.a[it.id].t}));
  const payload = JSON.stringify(
    {build: D.build, chunk: D.chunk, written: items.length,
     total: D.items.length, captions: items,
     finished: new Date().toISOString()}, null, 1);
  M.innerHTML = `<div class="done"><h2>Done &mdash; thank you</h2>
    <p>${items.length} of ${D.items.length} captions written.</p>
    <p>Click <b>Download</b> and send the file back. If the download is blocked, copy the
    text below into an email instead.</p>
    <p><button class="pri" id="dl">Download captions</button>
       <button id="cp">Copy to clipboard</button></p>
    <textarea class="out" readonly id="ta">${esc(payload)}</textarea></div>`;
  document.getElementById('dl').onclick = () => {
    const b = new Blob([payload], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `writing-${D.build}-chunk${D.chunk}.json`;
    document.body.appendChild(a); a.click(); a.remove();
  };
  document.getElementById('cp').onclick = () => {
    const ta = document.getElementById('ta'); ta.select();
    navigator.clipboard.writeText(payload).catch(()=>document.execCommand('copy'));
  };
}

document.getElementById('back').onclick = () => adv(-1);
document.getElementById('next').onclick = () => adv(1);
document.getElementById('skip').onclick = () => adv(1);
document.getElementById('exp').onclick = () => { S.i = D.items.length; save(); done(); };
render();
</script></body></html>
"""


def _h(*parts: str) -> int:
    return int(hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:12], 16)


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
    ap.add_argument("--n", type=int, default=250, help="captions to collect in total")
    ap.add_argument("--chunks", type=int, default=5,
                    help="split across this many standalone files, for parallel writers")
    ap.add_argument("--exclude", nargs="*", default=[
        "runs/human-eval/key.json", "runs/guess/legibility300_key.json"],
        help="key files whose images must not appear -- a writer must never have been "
             "shown one of these photographs WITH a register label")
    ap.add_argument("--out-prefix", default="WRITING")
    ap.add_argument("--key", default="runs/offdist/key.json")
    ap.add_argument("--max-dim", type=int, default=520)
    ap.add_argument("--quality", type=int, default=70)
    args = ap.parse_args()

    if args.n % (len(EMOTIONS) * args.chunks):
        raise SystemExit(f"--n {args.n} must divide evenly into {len(EMOTIONS)} registers "
                         f"x {args.chunks} chunks")
    cfg = load_config("data")
    images_dir = ROOT / cfg["paths"]["images_dir"]

    seen: set[str] = set()
    for f in args.exclude:
        p = ROOT / f
        if not p.exists():
            print(f"  note: {f} absent, nothing excluded from it")
            continue
        d = json.loads(p.read_text())
        entries = d.get("key", d)
        for v in entries.values():
            if isinstance(v, dict) and "image_id" in v:
                seen.add(v["image_id"])
    print(f"excluding {len(seen):,} images already shown with a label")

    pool = sorted(p.name for p in images_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg") and p.name not in seen)
    ranked = sorted(pool, key=lambda i: _h(SALT, i))
    if len(ranked) < args.n:
        raise SystemExit(f"only {len(ranked)} usable images for {args.n} captions")

    per_reg = args.n // len(EMOTIONS)
    items = []
    for r, reg in enumerate(EMOTIONS):
        for j in range(per_reg):
            img = ranked[r * per_reg + j]
            items.append({"id": hashlib.sha256(f"{SALT}|{img}|{reg}".encode()).hexdigest()[:12],
                          "image_id": img, "register": reg})
    # Deal each register evenly across the chunks, THEN shuffle within a chunk.
    #
    # Balancing per chunk rather than only overall matters because partial returns are the
    # normal case: five writers rarely all finish. If registers were spread at random, three
    # chunks coming back could leave one register with half the cells of another, and the
    # off-distribution accuracy would be a weighted average of five different sample sizes.
    # Dealt this way, ANY subset of returned chunks is still balanced.
    per_chunk = args.n // args.chunks
    per_reg_chunk = per_reg // args.chunks
    if per_reg % args.chunks:
        raise SystemExit(f"{per_reg} per register does not divide into {args.chunks} chunks")
    by_reg: dict[str, list[dict]] = {}
    for it in items:
        by_reg.setdefault(it["register"], []).append(it)
    chunks: list[list[dict]] = [[] for _ in range(args.chunks)]
    for reg, group in by_reg.items():
        for k, it in enumerate(group):
            chunks[k // per_reg_chunk].append(it)
    # Shuffle within each chunk so no writer gets a run of one tone, which would let them
    # settle into a formula rather than reading each photograph.
    for ch in chunks:
        ch.sort(key=lambda it: _h(SALT, "order", it["id"]))
    items = [it for ch in chunks for it in ch]

    build = hashlib.sha256("".join(it["id"] for it in items).encode()).hexdigest()[:10]
    print(f"embedding {args.n} photographs at <={args.max_dim}px q{args.quality} ...")

    for c in range(args.chunks):
        slice_ = items[c * per_chunk:(c + 1) * per_chunk]
        public = [{"id": it["id"], "register": it["register"],
                   "img": encode_image(images_dir / it["image_id"],
                                       args.max_dim, args.quality)} for it in slice_]
        data = {"build": build, "chunk": c + 1, "guide": dict(REGISTERS), "items": public}
        out = ROOT / f"{args.out_prefix}-{c+1}-of-{args.chunks}.html"
        out.write_text(
            PAGE.replace("__DATA__", json.dumps(data))
                .replace("__TITLE__", f"Write captions — part {c+1} of {args.chunks}"),
            encoding="utf-8")
        print(f"  {out.name}  {len(slice_)} items  {out.stat().st_size/1e6:.1f} MB  "
              f"{dict(Counter(i['register'] for i in slice_))}")

    kp = ROOT / args.key
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps({
        "build": build, "salt": SALT, "n": args.n, "chunks": args.chunks,
        "purpose": "off-distribution validation of the frozen register classifier; "
                   "250 hand-written captions, not written by Gemini",
        "excluded_from": args.exclude,
        "excluded_images": len(seen),
        "label_provenance": "the register is what the writer was ASKED for, shown beside "
                            "the photograph; no second annotator is needed because the "
                            "label precedes the text rather than being inferred from it",
        "key": {it["id"]: {"image_id": it["image_id"], "register": it["register"],
                           "chunk": i // per_chunk + 1}
                for i, it in enumerate(items)},
    }, indent=2))
    print(f"\nanswer key -> {rel(kp)}  (the pages never load it)")
    print(f"  {args.n} captions, {per_reg} per register, {args.chunks} writers x "
          f"{per_chunk} each")
    print(f"  collect the exports into runs/offdist/ and run scripts/score_offdist.py")


if __name__ == "__main__":
    main()
