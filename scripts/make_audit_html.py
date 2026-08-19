#!/usr/bin/env python
"""Build an interactive hand-audit page, and emit structured results.

`AUDIT.md` is readable but recording 200 judgements in it by hand is slow and the result is
prose someone then has to parse. This writes `AUDIT.html`: one image per screen with its five
registers beside it, judgements captured by clicking, autosaved to `localStorage`, and
exportable as JSON.

    uv run python scripts/make_audit_html.py
    open AUDIT.html

**Deliberately a local file, not a hosted page.** Images are referenced by relative path
from the repo root rather than embedded, so no Flickr8k photograph is copied anywhere —
consistent with `docs/data-attribution.md`, which commits to never redistributing them.
A `file://` page can also trigger a real download, which a sandboxed hosted page cannot.

Same sample as `AUDIT.md` when given the same `--seed`, so the two are interchangeable.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.grounding import caption_defects, load_factual_captions  # noqa: E402
from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.runtime import load_config  # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EmoCap hand audit</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e0ddd8;--card:#fff;
      --bad:#c0392b;--warn:#b8860b;--ok:#2d7a4f;--acc:#3b5bdb}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e6e3;--mut:#9a9a9a;
      --line:#2e3033;--card:#1f2124;--bad:#e06c5b;--warn:#d4a24c;--ok:#5cb87f;--acc:#7b93f0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--line);
  padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
.bar{flex:1;min-width:120px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);width:0%;transition:width .2s}
button{font:inherit;padding:6px 13px;border:1px solid var(--line);border-radius:7px;
  background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}
main{max-width:1180px;margin:0 auto;padding:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:18px;margin-bottom:22px}
.card.done{opacity:.55}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:12px}
.hd b{font-size:16px}.hd code{color:var(--mut);font-size:12px}
.split{display:grid;grid-template-columns:minmax(260px,340px) 1fr;gap:20px}
@media(max-width:820px){.split{grid-template-columns:1fr}}
img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.src{color:var(--mut);font-size:13px;margin:14px 0 6px;padding-left:9px;
  border-left:3px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:14px}
td{padding:7px 6px;border-top:1px solid var(--line);vertical-align:top}
td.reg{width:88px;font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.3px}
td.st{width:34px;text-align:center;color:var(--mut);font-variant-numeric:tabular-nums}
td.chk{width:150px;white-space:nowrap}
label.tog{display:inline-flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;
  color:var(--mut);margin-right:9px}
label.tog input{margin:0;cursor:pointer}
label.tog:has(input:checked){color:var(--bad);font-weight:600}
label.tog.r:has(input:checked){color:var(--warn)}
.flag{display:block;color:var(--warn);font-size:11.5px;margin-top:3px}
.foot{display:flex;gap:16px;align-items:center;margin-top:14px;padding-top:12px;
  border-top:1px solid var(--line);flex-wrap:wrap}
input[type=text]{flex:1;min-width:180px;font:inherit;padding:6px 9px;border-radius:7px;
  border:1px solid var(--line);background:var(--bg);color:var(--fg)}
.pill{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);
  color:var(--mut)}
.intro{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:18px;margin-bottom:26px}
.intro ol{margin:8px 0 0;padding-left:22px}.intro li{margin:5px 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.7px;color:var(--mut);
  margin:34px 0 14px;border-bottom:1px solid var(--line);padding-bottom:7px}
.sum td{border:none;padding:5px 8px 5px 0}
.sum input{width:70px}
.note{color:var(--mut);font-size:13px}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;
  padding:1px 5px;background:var(--bg)}
</style></head><body>
<header>
  <h1>EmoCap hand audit</h1>
  <span class="pill" id="prog">0 / __TOTAL__ images</span>
  <div class="bar"><i id="fill"></i></div>
  <button id="jump">Next unjudged</button>
  <button id="export" class="primary">Export JSON</button>
</header>
<main>
<div class="intro">
  <b>For each image, open it and ask four things.</b>
  <ol>
    <li><b>Is every caption true of the photograph?</b> Tick <i>contradicts</i> for anything
        invented, wrong, or not visible.</li>
    <li><b>Would you guess the intended register from the caption alone?</b> Tick
        <i>register fails</i> if not.</li>
    <li><b>Are the five meaningfully different</b>, or one sentence with a swapped adjective?</li>
    <li><b>Does it read like a person wrote it</b>, or like a template?</li>
  </ol>
  <p class="note"><b>strain</b> is the <i>generator's own</i> claim about register fit —
  0 natural, 1 strained, 2 no honest reading exists. Whether you agree is part of the audit:
  it is an instrument being validated, not a fact. Tick <i>strain wrong</i> when you disagree.</p>
  <p class="note">Work autosaves in this browser. Nothing leaves your machine until you press
  <b>Export JSON</b>. Untouched captions count as fine, so you only click on problems.</p>
</div>
<div id="app"></div>
<h2>Summary — fill in when done</h2>
<div class="card">
<table class="sum">
<tr><td>Images where the five felt interchangeable</td><td id="autoInter" class="note">—</td></tr>
<tr><td>Captions contradicting their image</td><td id="autoContra" class="note">—</td></tr>
<tr><td>Registers you could not guess</td><td id="autoFail" class="note">—</td></tr>
<tr><td>Cells where you disagreed with <code>strain</code></td><td id="autoStrain" class="note">—</td></tr>
</table>
<p><b>Is this corpus good enough to train on?</b><br>
<input type="text" id="s_verdict" placeholder="yes / no / qualified — and why"></p>
<p><b>The single biggest weakness</b><br>
<input type="text" id="s_weak" placeholder=""></p>
<p><b>Anything that should change before the preregistration is tagged</b><br>
<input type="text" id="s_change" placeholder=""></p>
<div class="foot"><button id="export2" class="primary">Export JSON</button>
<span class="note">saves <code>emocap-audit-results.json</code> — hand that file back</span></div>
</div>
</main>
<script>
const DATA = __DATA__;
const KEY = 'emocap-audit-' + DATA.seed;
let S = JSON.parse(localStorage.getItem(KEY) || '{}');
S.images = S.images || {};
const save = () => localStorage.setItem(KEY, JSON.stringify(S));
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function rec(id){ S.images[id] = S.images[id] || {caps:{}, interchangeable:false, note:'', seen:false}; return S.images[id]; }

function render(){
  const app = document.getElementById('app');
  let h = '';
  for (const part of ['A','B']){
    const items = DATA.items.filter(x => x.part === part);
    if (!items.length) continue;
    h += part === 'A'
      ? `<h2>Part A — the audit · ${items.length} images · ${items.length*5} cells · unbiased random sample</h2>`
      : `<h2>Part B — flagged examples · ${items.length} images · NOT part of the 200</h2>`;
    if (part === 'B') h += `<p class="note">Selected <i>because</i> a detector flagged them, so
      unrepresentative by construction. Here so you can see what the corpus-wide 1.93% defect
      rate looks like — and judge whether the detector was <b>right</b>.</p>`;
    items.forEach((it, i) => {
      const r = rec(it.image_id);
      h += `<div class="card ${r.seen?'done':''}" id="c_${it.image_id}">
        <div class="hd"><b>${part}${i+1}/${items.length}</b><code>${esc(it.image_id)}</code></div>
        <div class="split"><div><img loading="lazy" src="${esc(it.rel)}" alt=""></div><div>`;
      it.sources.forEach(src => {
        h += `<div class="src">source ${src.idx}: ${esc(src.text)}</div><table>`;
        src.rows.forEach(row => {
          const c = r.caps[row.key] || {};
          h += `<tr><td class="reg">${row.reg}</td>
            <td>${esc(row.text)}${row.flags.length?`<span class="flag">flagged — ${esc(row.flags.join('; '))}</span>`:''}</td>
            <td class="st">${row.strain}</td>
            <td class="chk">
              <label class="tog"><input type="checkbox" data-k="${row.key}" data-f="contradicts" ${c.contradicts?'checked':''}>contradicts</label>
              <label class="tog r"><input type="checkbox" data-k="${row.key}" data-f="register_fails" ${c.register_fails?'checked':''}>register fails</label>
              <label class="tog r"><input type="checkbox" data-k="${row.key}" data-f="strain_wrong" ${c.strain_wrong?'checked':''}>strain wrong</label>
            </td></tr>`;
        });
        h += `</table>`;
      });
      h += `</div></div><div class="foot">
        <label class="tog"><input type="checkbox" data-img="${it.image_id}" data-f="interchangeable" ${r.interchangeable?'checked':''}>the five are interchangeable</label>
        <input type="text" data-img="${it.image_id}" data-f="note" value="${esc(r.note||'')}" placeholder="${part==='B'?'was the detector right?':'note (optional)'}">
        <button data-done="${it.image_id}">${r.seen?'judged ✓':'mark judged'}</button>
      </div></div>`;
    });
  }
  app.innerHTML = h;
  bind(); tally();
}

function bind(){
  document.querySelectorAll('input[type=checkbox][data-k]').forEach(el => el.onchange = e => {
    const r = rec(e.target.closest('.card').id.slice(2));
    r.caps[e.target.dataset.k] = r.caps[e.target.dataset.k] || {};
    r.caps[e.target.dataset.k][e.target.dataset.f] = e.target.checked;
    save(); tally();
  });
  document.querySelectorAll('[data-img]').forEach(el => {
    const h = e => { const r = rec(e.target.dataset.img);
      r[e.target.dataset.f] = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
      save(); tally(); };
    el.onchange = h; if (el.type === 'text') el.oninput = h;
  });
  document.querySelectorAll('[data-done]').forEach(el => el.onclick = e => {
    const id = e.target.dataset.done, r = rec(id);
    r.seen = !r.seen; save();
    document.getElementById('c_' + id).classList.toggle('done', r.seen);
    e.target.textContent = r.seen ? 'judged ✓' : 'mark judged';
    tally();
  });
  ['s_verdict','s_weak','s_change'].forEach(k => {
    const el = document.getElementById(k);
    el.value = (S.summary || {})[k] || '';
    el.oninput = () => { S.summary = S.summary || {}; S.summary[k] = el.value; save(); };
  });
}

function tally(){
  const A = DATA.items.filter(x => x.part === 'A').map(x => x.image_id);
  let seen = 0, contra = 0, fail = 0, strain = 0, inter = 0;
  A.forEach(id => { const r = S.images[id]; if (!r) return;
    if (r.seen) seen++; if (r.interchangeable) inter++;
    Object.values(r.caps || {}).forEach(c => {
      if (c.contradicts) contra++; if (c.register_fails) fail++; if (c.strain_wrong) strain++; });
  });
  document.getElementById('prog').textContent = `${seen} / ${A.length} images judged`;
  document.getElementById('fill').style.width = (100*seen/A.length) + '%';
  document.getElementById('autoInter').textContent = `${inter} / ${A.length}`;
  document.getElementById('autoContra').textContent = `${contra} / ${A.length*5}`;
  document.getElementById('autoFail').textContent = `${fail} / ${A.length*5}`;
  document.getElementById('autoStrain').textContent = `${strain}`;
}

document.getElementById('jump').onclick = () => {
  const next = DATA.items.find(x => !(S.images[x.image_id] || {}).seen);
  if (next) document.getElementById('c_' + next.image_id).scrollIntoView({behavior:'smooth', block:'start'});
};

function exportJSON(){
  const A = DATA.items.filter(x => x.part === 'A');
  const out = {
    tool:'make_audit_html.py', seed:DATA.seed, corpus:DATA.corpus,
    exported_at:new Date().toISOString(),
    counts:{
      part_a_images:A.length, part_a_cells:A.length*5,
      images_judged:A.filter(x=>(S.images[x.image_id]||{}).seen).length,
      contradicts:0, register_fails:0, strain_wrong:0, interchangeable:0},
    part_a:[], part_b:[], summary:S.summary || {}
  };
  DATA.items.forEach(it => {
    const r = S.images[it.image_id] || {caps:{}};
    const problems = Object.entries(r.caps || {})
      .filter(([,v]) => v.contradicts || v.register_fails || v.strain_wrong)
      .map(([k,v]) => ({cell:k, ...v}));
    if (it.part === 'A'){
      problems.forEach(p => { if(p.contradicts) out.counts.contradicts++;
        if(p.register_fails) out.counts.register_fails++; if(p.strain_wrong) out.counts.strain_wrong++; });
      if (r.interchangeable) out.counts.interchangeable++;
      out.part_a.push({image_id:it.image_id, judged:!!r.seen,
        interchangeable:!!r.interchangeable, note:r.note||'', problems});
    } else {
      out.part_b.push({image_id:it.image_id, judged:!!r.seen,
        detector_note:r.note||'', problems});
    }
  });
  const b = new Blob([JSON.stringify(out,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'emocap-audit-results.json'; a.click();
}
document.getElementById('export').onclick = exportJSON;
document.getElementById('export2').onclick = exportJSON;
render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=40,
                    help="images sampled; ONE source caption is drawn per image, so cells = "
                         "images x 5 registers. 40 -> the 200 cells configs/data.yaml sets "
                         "as audit_sample.")
    ap.add_argument("--flagged", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--store", default=None,
                    help="caption store to audit; defaults to paths.raw_generations. Needed "
                         "to audit an alternative arm (a different generator or prompt) "
                         "rather than the production corpus.")
    ap.add_argument("--out", default="AUDIT.html")
    args = ap.parse_args()

    cfg = load_config("data")
    store = Path(args.store) if args.store else ROOT / cfg["paths"]["raw_generations"]
    if not store.is_absolute():
        store = ROOT / store
    recs = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
    print(f"auditing {store.name}")
    by_image: dict[str, list[dict]] = {}
    for r in recs:
        by_image.setdefault(r["image_id"], []).append(r)
    for v in by_image.values():
        v.sort(key=lambda r: r["caption_idx"])

    try:
        factual = load_factual_captions(
            ROOT / "archive/v1-pilot/data/v1_moondream_factual_captions.csv.gz")
    except FileNotFoundError:
        factual = {}

    def flags(rec: dict, reg: str) -> list[str]:
        f = []
        if reg in (rec.get("rejected") or {}):
            f.append(rec["rejected"][reg])
        f += caption_defects(str(rec["captions"].get(reg, "")),
                             source_caption=rec["source_caption"],
                             factual_caption=factual.get(rec["image_id"]))
        return f

    rng = random.Random(args.seed)
    rng_pick = random.Random(args.seed + 1)   # separate stream so the image sample is stable
    all_images = sorted(by_image)
    sample = rng.sample(all_images, args.images)
    pool = [i for i in all_images if i not in set(sample)
            and any(flags(r, e) for r in by_image[i] for e in EMOTIONS)]
    rng.shuffle(pool)
    flagged = pool[: args.flagged]

    imgdir = Path(cfg["paths"]["images_dir"])
    items = []
    for part, group in (("A", sample), ("B", flagged)):
        for img in group:
            # ONE source caption per image, chosen deterministically from the seed. Using
            # all five would make each image 25 cells and the sample 1,000 -- five times the
            # 200 the preregistration specifies -- for far less image diversity per caption
            # read.
            chosen = rng_pick.choice(by_image[img])
            sources = []
            for rec_ in [chosen]:
                rows = [{"reg": e, "key": f"{rec_['caption_idx']}:{e}",
                         "text": rec_["captions"].get(e, "—"),
                         "strain": (rec_.get("strain") or {}).get(e, "–"),
                         "flags": flags(rec_, e)} for e in EMOTIONS]
                if part == "B":
                    rows = [r for r in rows if r["flags"]] or rows
                sources.append({"idx": rec_["caption_idx"],
                                "text": rec_["source_caption"].strip(), "rows": rows})
            items.append({"part": part, "image_id": img,
                          "rel": str(imgdir / img), "sources": sources})

    data = {"seed": args.seed,
            "corpus": {"records": len(recs), "images": len(all_images),
                       "cells": sum(len(r["captions"]) for r in recs)},
            "items": items}
    out = ROOT / args.out
    out.write_text(
        PAGE.replace("__DATA__", json.dumps(data))
            .replace("__TOTAL__", str(len(sample))),
        encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  Part A: {len(sample)} images / {len(sample) * 5} cells")
    print(f"  Part B: {len(flagged)} flagged images")
    print(f"\n  open {out}")


if __name__ == "__main__":
    main()
