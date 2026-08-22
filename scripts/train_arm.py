#!/usr/bin/env python
"""Train one (arm, fold) and decode its held-out cells. The unit of the 36-run design.

    uv run python scripts/train_arm.py --arm S_unpaired --fold 0
    uv run python scripts/train_arm.py --arm S_unpaired --fold 0 --negative-control

One process per run, deliberately. A single script that loops all 36 cannot be resumed,
cannot be split across Kaggle sessions, and turns one crash into a lost day. Each run
writes its own directory with a manifest, the decoded captions, and the trainable
parameter count that §2 requires matched across arms.

Nothing here is per-arm tunable: every hyperparameter comes from `configs/model.yaml` and
every decode setting from the frozen `DecodeConfig` in the lock. The arm name selects a
data file and nothing else, which is the property that makes the comparison mean anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.decode.beam import DecodeConfig  # noqa: E402
from emocap.features.clip import load_features  # noqa: E402
from emocap.models.clipcap import ClipCap, ClipCapConfig  # noqa: E402
from emocap.models.dataset import ArmCells, collate, load_arm, make_split  # noqa: E402
from emocap.runtime import Manifest, load_config, seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--negative-control", action="store_true",
                    help="shuffle emotion labels within the arm (the registered control)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=None,
                    help="where arms/ and features/ live; defaults to the repo's data/")
    ap.add_argument("--out-root", default="runs/arms")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override ONLY for smoke tests -- changes the registered budget")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: cap training cells")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    cfg = load_config("model")
    opt_cfg = cfg["optim"]
    root = Path(args.data_root) if args.data_root else ROOT / "data"

    lock = load_config("prereg.lock")
    dc = DecodeConfig(**dict(lock["decode"]))

    epochs = args.epochs or opt_cfg["epochs"]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    seed_everything(args.seed)

    arm = load_arm(root / "arms" / f"{args.arm}.jsonl")
    if args.negative_control:
        # Shuffled WITHIN the arm: the marginal label distribution is untouched, so the
        # control's accuracy stays comparable to the arm's. Only the image-emotion
        # pairing is destroyed, which is the thing the control is supposed to test.
        arm = arm.shuffled_labels(seed=args.seed)
    train, held = make_split(arm, args.fold)
    if args.limit:
        train = ArmCells(train.cells[:args.limit])
    store = load_features(root / "features")

    tok = AutoTokenizer.from_pretrained(cfg["decoder"]["model"])
    tok.pad_token = tok.eos_token
    model = ClipCap(ClipCapConfig.from_config(cfg)).to(device)
    n_params = model.trainable_parameters()

    tag = f"{args.arm}-f{args.fold}" + ("-nc" if args.negative_control else "")
    run_dir = ROOT / args.out_root / tag
    man = Manifest(run_id=tag, stage="05_train_arm", config={
        "arm": args.arm, "fold": args.fold, "negative_control": args.negative_control,
        "seed": args.seed, "epochs": epochs, "device": device,
        "train_cells": len(train), "held_out_cells": len(held),
        "trainable_parameters": n_params, "model": cfg["decoder"]["model"],
        "decode": dc.as_dict()})
    man.save(run_dir)
    print(f"{tag}  train {len(train):,}  held-out {len(held):,}  "
          f"trainable {n_params:,}  device {device}", flush=True)

    max_len = cfg["decoder"]["max_seq_len"] - model.total_prefix
    loader = DataLoader(train.cells, batch_size=opt_cfg["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, store, tok, max_len=max_len))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=opt_cfg["lr"],
                              weight_decay=opt_cfg["weight_decay"])
    total_steps = max(1, len(loader) * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=opt_cfg["lr"], total_steps=total_steps,
        pct_start=min(0.3, opt_cfg["warmup_steps"] / total_steps),
        anneal_strategy="cos")

    # Mixed precision on CUDA only. `configs/model.yaml` asks for fp16; MPS autocast is
    # not reliable for this stack and the local machine is only ever used for smoke tests,
    # so the flag is honoured where it is real and ignored where it is not -- rather than
    # silently ignored everywhere, which is what this code did before.
    use_amp = device == "cuda" and opt_cfg.get("amp") == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  amp {'fp16' if use_amp else 'off'}  steps/epoch {len(loader):,}", flush=True)

    t0 = time.time()
    model.train()
    losses: list[float] = []
    for ep in range(epochs):
        run_loss, n = 0.0, 0
        for feats, emos, ids, mask in loader:
            optim.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(feats.to(device), emos.to(device), ids.to(device),
                             mask.to(device))
            scaler.scale(loss).backward()
            # Unscale before clipping, or the clip threshold applies to scaled gradients
            # and does nothing at fp16's loss scale.
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(trainable, opt_cfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            n += 1
        losses.append(round(run_loss / max(1, n), 4))
        print(f"  epoch {ep+1}/{epochs}  loss {losses[-1]:.4f}  "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    # EVERY held-out cell is decoded -- prereg-v2 `evaluation.eval_subsample: none`.
    model.eval()
    preds: list[dict] = []
    bs = 64
    for s in range(0, len(held.cells), bs):
        chunk = held.cells[s:s + bs]
        feats, emos, _, _ = collate(chunk, store, tok, max_len=max_len)
        caps = model.generate(feats.to(device), emos.to(device), tok, config=dc)
        for c, cap in zip(chunk, caps):
            preds.append({"image_id": c["image_id"], "emotion": c["emotion"],
                          "reference": c["text"], "generated": cap, "fold": args.fold})
        if s % (bs * 20) == 0:
            print(f"  decoded {s + len(chunk):,}/{len(held.cells):,}  "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in preds))
    stats = {"losses": losses, "wall_minutes": round((time.time() - t0) / 60, 1),
             "decoded": len(preds), "trainable_parameters": n_params,
             "empty_captions": sum(1 for p in preds if not p["generated"].strip())}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    man.finalise(run_dir)
    print(f"\n{tag}  decoded {len(preds):,}  empty {stats['empty_captions']}  "
          f"wall {stats['wall_minutes']} min  -> {run_dir}")


if __name__ == "__main__":
    main()
