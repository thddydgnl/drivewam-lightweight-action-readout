"""Train PhysicalAI student heads on cached features and predict dev/final rows (batch one).

Protocol (pre-registration §5): 40 epochs fixed, AdamW 3e-4, wd 0.01, batch 32, warmup 5% +
cosine, last checkpoint, no selection on any evaluation set. One process trains the given
seeds sequentially for one (train set, variant, phase). `--frame ego` (default, change 9): the
head predicts chunk-start ego-frame deltas, rotated into the clip frame by the heading from the
registered GT actions; `--frame clip` is the originally registered parametrization (ablation).
Pre-registration 2026-11-01: `--head ego` trains the ego-only control (no world features read);
`--eval-tag T` predicts eval features from caches/pai-features/<set>-T (e.g. other noise seeds)
into pred-<set>-T.npz of the already trained heads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
from pai_heads import PaiHead, PaiEgoHead, loss_fn, heading_before, to_clip_frame   # noqa: E402

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for b in iter(lambda: s.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()

def source(set_name, variant, phase):
    """Feature folder (and clip filter) for a (set, variant, phase); 4k first rows live in train20k."""
    if set_name == "train4k" and variant == "normal" and phase == "first":
        return "train20k-normal", ROOT / "splits/pai-train-4k.txt"
    if set_name == "train4k" and variant == "normal":
        return "train4k-ct", None
    if set_name == "train20k" and variant == "normal" and phase == "current":
        return "train20k-current", None
    return f"{set_name}-{variant}", None

def mapped(path, name):
    """Zero-copy view of one safetensors tensor (page cache shared by all processes)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        info = json.loads(f.read(n))[name]
    assert info["dtype"] == "BF16", info
    start, end = info["data_offsets"]
    arr = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + start, shape=tuple(info["shape"]))
    return torch.from_numpy(arr).view(torch.bfloat16)

class Rows:
    """World features as memory-mapped rows; `take(idx)` gathers a contiguous bf16 batch."""
    def __init__(self, views, index):
        self.views, self.index = views, index                    # index[i] = (view id, local row)
    def __len__(self):
        return len(self.index)
    def take(self, idx):
        return torch.stack([self.views[self.index[i][0]][self.index[i][1]] for i in idx.tolist()])

def load(set_name, variant, phase, folder=None):
    folder, filt = (folder, None) if folder else source(set_name, variant, phase)
    root = ROOT / "caches/pai-features" / folder
    keep = set(filt.read_text().split()) if filt else None
    if keep is None:
        for s in [0, 1]:
            assert json.loads((root / f"status-shard{s}.json").read_text()).get("stage") == "verified", (root, s)
    files = sorted(root.glob("shard*/*.safetensors"))
    picks, n = [], 0
    for f in files:
        meta = json.loads(Path(str(f) + ".json").read_text())
        idx = [j for j, r in enumerate(meta["rows"]) if keep is None or r["clip"] in keep]
        if idx:
            picks.append((f, meta, idx)); n += len(idx)
    split = {"train4k": "pai-train-4k.txt", "train20k": "pai-train-20k.txt", "dev": "pai-dev-1k.txt",
             "final": "pai-final-1k.txt"}[set_name]
    required = keep if keep is not None else set((ROOT / "splits" / split).read_text().split())
    failed = {q.stem for q in (ROOT / "logs/pai-process-errors").glob("*.txt")}
    present = {meta["rows"][j]["clip"] for _, meta, idx in picks for j in idx}
    assert not (required - failed - present), (folder, len(required - failed - present))
    views, index = [], []
    ego = torch.empty((n, 5)); target = torch.empty((n, 40, 3)); rows = []
    i = 0
    for f, meta, idx in picks:
        sel = torch.tensor(idx); k = len(idx)
        views.append(mapped(str(f), f"dense_{phase}"))
        index += [(len(views) - 1, j) for j in idx]
        with safe_open(str(f), framework="pt", device="cpu") as s:
            ego[i:i + k] = s.get_tensor("ego").index_select(0, sel)
            target[i:i + k] = s.get_tensor("target").index_select(0, sel)
        rows += [meta["rows"][j] for j in idx]; i += k
    by_clip = {}
    for r_i, r in enumerate(rows):
        by_clip.setdefault(r["clip"], {})[r["chunk"]] = r_i
    theta = torch.empty(n, dtype=torch.float64)
    for c, chunks in by_clip.items():
        order = sorted(chunks)
        assert order == list(range(1, len(order) + 1)), (c, order)
        prev = []
        for k in order:
            theta[chunks[k]] = heading_before(prev)
            prev.append(target[chunks[k]].numpy())
    return Rows(views, index), ego, target, rows, theta.float()

def head_out(model, w, e, th, frame):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(w, e)
    return to_clip_frame(out, th) if frame == "ego" else out

@torch.no_grad()
def predict(folder, model, mu, sd, evals, device, frame):
    """Batch-one predictions (as deployed) for every eval set not yet predicted (clip frame)."""
    model.eval()
    done = {}
    for es, (ew, ee, et, er, eth) in evals.items():
        path = folder / f"pred-{es}.npz"
        if path.exists():
            done[es] = str(path); continue
        outs = []
        ego_only = isinstance(model, PaiEgoHead)
        for i in range(len(ew)):
            w = None if ego_only else ew.take(torch.tensor([i])).to(device)
            o = head_out(model, w, ((ee[i:i + 1] - mu) / sd).to(device), eth[i:i + 1].to(device), frame)
            outs.append(o.float().cpu()[0])
        np.savez(path, normalized=torch.stack(outs).numpy(), clips=np.asarray([r["clip"] for r in er]),
                 chunks=np.asarray([r["chunk"] for r in er]))
        done[es] = str(path)
    return done

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)            # train4k | train20k
    p.add_argument("--variant", default="normal")
    p.add_argument("--phase", default="first")
    p.add_argument("--seeds", default="17")
    p.add_argument("--eval-sets", default="")           # comma list: dev,final (empty = train only)
    p.add_argument("--predict", action="store_true", help="only predict eval sets with trained heads")
    p.add_argument("--frame", default="ego", choices=["ego", "clip"])
    p.add_argument("--parallel-seeds", action="store_true", help="one process per seed (shared page cache)")
    p.add_argument("--run", default="pai_heads_2026-10-04")
    p.add_argument("--head", default="sf3", choices=["sf3", "ego"])
    p.add_argument("--eval-tag", default=None, help="eval features from <set>-<tag>, predictions pred-<set>-<tag>.npz")
    a = p.parse_args()
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda:0"); torch.set_num_threads(8)
    if a.eval_tag:
        evals = {f"{es}-{a.eval_tag}": load(es, a.variant, a.phase, folder=f"{es}-{a.eval_tag}") for es in a.eval_sets.split(",") if es}
    else:
        evals = {es: load(es, a.variant, a.phase) for es in a.eval_sets.split(",") if es}
    Head = PaiEgoHead if a.head == "ego" else PaiHead
    kind = "egoonly" if a.head == "ego" else a.phase
    out_root = ROOT / "runs" / a.run
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.parallel_seeds and len(seeds) > 1:
        import subprocess
        args = [x for x in sys.argv[1:] if x != "--parallel-seeds"]
        k = args.index("--seeds")
        procs = [subprocess.Popen([sys.executable, "-u", __file__] + args[:k] + ["--seeds", str(sd)] + args[k + 2:])
                 for sd in seeds]
        codes = [q.wait() for q in procs]
        print(json.dumps({"parallel_seeds": seeds, "exit_codes": codes}), flush=True)
        sys.exit(max(codes))
    if a.predict:
        for seed in seeds:
            folder = out_root / f"{a.train}-{a.variant}-{kind}{'-clipframe' if a.frame == 'clip' else ''}-seed{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            assert summary.get("frame", "ego") == a.frame
            model = Head().to(device)
            model.load_state_dict(load_file(str(folder / "epoch-040.safetensors")))
            st = load_file(str(folder / "stats.safetensors"))
            summary["predictions"] = {**summary.get("predictions", {}),
                                      **predict(folder, model, st["ego_mean"], st["ego_std"], evals, device, a.frame)}
            (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"predicted": folder.name, "sets": list(summary["predictions"])}), flush=True)
        sys.exit(0)
    world, ego, target, _, theta = load(a.train, a.variant, a.phase)
    mu, sd = ego.mean(0), ego.std(0, unbiased=False).clamp_min(1e-5)
    n = len(world)
    for seed in seeds:
        name = f"{a.train}-{a.variant}-{kind}{'-clipframe' if a.frame == 'clip' else ''}-seed{seed}"
        folder = out_root / name
        if (folder / "summary.json").exists():
            continue
        folder.mkdir(exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = Head().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999))
        total = 40 * math.ceil(n / 32); warm = int(total * 0.05); step = 0; hist = []; t0 = time.monotonic()
        for epoch in range(1, 41):
            model.train()
            order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 100003 * epoch))
            acc = 0.0
            for off in range(0, n, 32):
                idx = order[off:off + 32]
                w = None if a.head == "ego" else world.take(idx).to(device, non_blocking=True)
                e = ((ego.index_select(0, idx) - mu) / sd).to(device)
                y = target.index_select(0, idx).to(device)
                th = theta.index_select(0, idx).to(device)
                step += 1
                f = step / max(1, warm) if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
                for g in opt.param_groups:
                    g["lr"] = 3e-4 * f
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(head_out(model, w, e, th, a.frame), y)
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                acc += float(loss.detach()) * len(idx)
            hist.append({"epoch": epoch, "train_loss": acc / n})
            print(json.dumps({"run": name, "epoch": epoch, "train_loss": round(acc / n, 5),
                              "elapsed_s": round(time.monotonic() - t0)}), flush=True)
        weights = folder / "epoch-040.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights))
        save_file({"ego_mean": mu.contiguous(), "ego_std": sd.contiguous()}, str(folder / "stats.safetensors"))
        preds = predict(folder, model, mu, sd, evals, device, a.frame)
        (folder / "history.json").write_text(json.dumps(hist) + "\n")
        (folder / "summary.json").write_text(json.dumps({"name": name, "frame": a.frame, "train_rows": n, "epochs": 40, "steps": step,
            "checkpoint_sha256": sha(weights), "predictions": preds, "elapsed_s": time.monotonic() - t0,
            "utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        print(json.dumps({"done": name, "final_loss": hist[-1]["train_loss"]}), flush=True)
