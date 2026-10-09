"""Round-3 head training and navtest-dev prediction (pre-registration §6).

One process = one phase and one data size; trains R and A for each requested
seed. Only the requested phase's [10,20,30] features are loaded. Protocol is
v2 (40 fixed epochs, last checkpoint, batch-1 prediction) with FP32 outputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from lightweight_heads import normalize_deltas, parameter_count
from heads_v3 import RegressionHead, AnchorHead, kmeans_anchors, nearest_anchor

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
PHASE = {"current": 0, "first": 1, "terminal": 2}
SIX_TO_THREE = torch.tensor([1, 3, 5])
EPOCHS = 40

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda: s.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)

def read_rows(shards, phase, six_layers):
    """shards: list of (path, tokens). Returns {token: (dense[3,112,3072] bf16, ego[20])}."""
    rows = {}
    for path, tokens in shards:
        with safe_open(path, framework="pt", device="cpu") as f:
            dense = f.get_slice("dense")[:, phase:phase + 1][:, 0]
            if six_layers:
                dense = dense.index_select(1, SIX_TO_THREE)
            ego = f.get_tensor("ego")
        for i, t in enumerate(tokens):
            rows[t] = (dense[i].clone(), ego[i].clone())
    return rows

def labels(manifest, tokens):
    rows = {r["scene_token"]: r for r in map(json.loads, Path(manifest).read_text().splitlines())}
    poses, deltas = [], []
    for t in tokens:
        with np.load(rows[t]["label_path"], allow_pickle=False) as d:
            poses.append(torch.from_numpy(d["future_poses_4s"].copy())); deltas.append(torch.from_numpy(d["action_deltas_4s"].copy()))
    return torch.stack(poses), normalize_deltas(torch.stack(deltas))

def pilot_sources():
    cache = ROOT / "caches/features/world_features_pilot_2026-10-01"
    lock = json.loads((cache / "FEATURE_LOCK.json").read_text())
    shards = []
    for record in lock["shards"]:
        assert Path(record["path"]).stat().st_size == record["bytes"]
        shards.append((record["path"], record["tokens"]))
    return shards

def x1_sources():
    cache = ROOT / "caches/features/x1_features_2026-10-03"
    lock = json.loads((cache / "FEATURE_LOCK.json").read_text())
    assert lock["status"] == "x1_dense_features_verified"
    shards = []
    for r in lock["shards"]:
        assert sha(r["path"]) == r["sha256"], r["path"]
        shards.append((r["path"], r["tokens"]))
    return shards

def dev_sources():
    out = ROOT / "runs/navtest_dev_quality_2026-10-02"
    shards = []
    for receipt in sorted(out.glob("shard*/*.json")):
        report = json.loads(receipt.read_text())
        assert Path(report["features_path"]).is_file()
        shards.append((report["features_path"], report["tokens"]))
    return shards

def stack(rows, tokens):
    return torch.stack([rows[t][0] for t in tokens]), torch.stack([rows[t][1] for t in tokens])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["first", "terminal"], required=True)
    p.add_argument("--data", choices=["4k", "20k", "full"], required=True)
    p.add_argument("--kinds", default="R,A")
    p.add_argument("--seeds", default="17,29")
    p.add_argument("--run", default="heads_v3_2026-10-03")
    a = p.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    phase = PHASE[a.phase]
    out_root = ROOT / "runs" / a.run
    out_root.mkdir(parents=True, exist_ok=True)

    pilot_index = [json.loads(l) for l in (ROOT / "caches/features/world_features_pilot_2026-10-01/index.jsonl").read_text().splitlines()]
    train_tokens = [r["scene_token"] for r in pilot_index if r["split"] == "head_train"]
    val_tokens = [r["scene_token"] for r in pilot_index if r["split"] == "head_val"]
    rows = read_rows(pilot_sources(), phase, six_layers=True)
    train_world, train_ego = stack(rows, train_tokens)
    val_world, val_ego = stack(rows, val_tokens)
    del rows
    train_poses, train_target = labels(ROOT / "splits/pilot-manifest.jsonl", train_tokens)
    val_poses, _ = labels(ROOT / "splits/pilot-manifest.jsonl", val_tokens)
    if a.data == "full":
        # Pre-allocate once (~191 GB BF16) and fill in place; never hold a second copy.
        x1_tokens = (ROOT / "splits/x1-head-train-tokens.txt").read_text().splitlines()
        full_lock = json.loads((ROOT / "caches/features/full_first_2026-10-03/FEATURE_LOCK.json").read_text())
        assert full_lock["status"] == "full_first_features_verified" and phase == 1
        full_tokens = (ROOT / "splits/full-extra-tokens.txt").read_text().splitlines()
        all_tokens = train_tokens + x1_tokens + full_tokens
        assert len(all_tokens) == len(set(all_tokens))
        position = {t: i for i, t in enumerate(all_tokens)}
        world = torch.empty((len(all_tokens), 3, 112, 3072), dtype=torch.bfloat16)
        ego_all = torch.empty((len(all_tokens), 20), dtype=train_ego.dtype)
        world[:len(train_tokens)] = train_world; ego_all[:len(train_tokens)] = train_ego
        del train_world
        def fill(shards, slicer):
            for path, toks in shards:
                with safe_open(path, framework="pt", device="cpu") as f:
                    dense = slicer(f.get_slice("dense")); e = f.get_tensor("ego")
                for i, t in enumerate(toks):
                    world[position[t]] = dense[i]; ego_all[position[t]] = e[i]
        fill(x1_sources(), lambda s: s[:, 1:2][:, 0])
        shards = []
        for r in full_lock["shards"]:
            assert sha(r["path"]) == r["sha256"], r["path"]
            shards.append((r["path"], r["tokens"]))
        fill(shards, lambda s: s[:])
        po1, ta1 = labels(ROOT / "splits/x1-manifest.jsonl", x1_tokens)
        po2, ta2 = labels(ROOT / "splits/full-extra-manifest.jsonl", full_tokens)
        train_world, train_ego = world, ego_all
        train_poses, train_target = torch.cat([train_poses, po1, po2]), torch.cat([train_target, ta1, ta2])
        train_tokens = all_tokens
    if a.data == "20k":
        x1_tokens = (ROOT / "splits/x1-head-train-tokens.txt").read_text().splitlines()
        rows = read_rows(x1_sources(), phase, six_layers=False)
        w, e = stack(rows, x1_tokens); del rows
        po, ta = labels(ROOT / "splits/x1-manifest.jsonl", x1_tokens)
        train_world, train_ego = torch.cat([train_world, w]), torch.cat([train_ego, e])
        train_poses, train_target = torch.cat([train_poses, po]), torch.cat([train_target, ta])
        train_tokens = train_tokens + x1_tokens
        del w, e
    dev_tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
    rows = read_rows(dev_sources(), phase, six_layers=True)
    dev_world, dev_ego = stack(rows, dev_tokens); del rows
    n = len(train_tokens)
    assert train_world.shape[1:] == (3, 112, 3072) and len(train_world) == n == len(train_poses)
    ego_mean, ego_std = train_ego.mean(0), train_ego.std(0, unbiased=False).clamp_min(1e-5)
    norm = lambda e: (e - ego_mean) / ego_std
    if "A" in a.kinds.split(","):
        anchors = kmeans_anchors(train_poses)
        anchor_index = nearest_anchor(anchors, train_poses)
    else:                                   # regression only: anchors unused
        anchors = torch.zeros(1, 8, 3)
        anchor_index = torch.zeros(len(train_poses), dtype=torch.long)
    save_file({"ego_mean": ego_mean.contiguous(), "ego_std": ego_std.contiguous(), "anchors": anchors.contiguous()},
              str(out_root / f"stats-{a.data}-{a.phase}.safetensors"))

    for seed in [int(s) for s in a.seeds.split(",")]:
        for kind in a.kinds.split(","):
            folder = out_root / f"{a.data}-{kind}-{a.phase}-seed{seed}"
            if (folder / "summary.json").exists():
                continue
            folder.mkdir(exist_ok=True)
            torch.manual_seed(seed); np.random.seed(seed)
            model = (RegressionHead() if kind == "R" else AnchorHead(anchors)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, betas=(.9, .999))
            steps_per_epoch = math.ceil(n / 32)
            total, step, history, start = EPOCHS * steps_per_epoch, 0, [], time.monotonic()
            warmup = int(total * .05)
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 100003 * epoch))
                loss_sum = 0.
                for off in range(0, n, 32):
                    idx = order[off:off + 32]
                    world = train_world.index_select(0, idx).to(device, non_blocking=True)
                    ego = norm(train_ego.index_select(0, idx)).to(device)
                    poses = train_poses.index_select(0, idx).to(device)
                    target = train_target.index_select(0, idx).to(device)
                    step += 1
                    factor = step / max(1, warmup) if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
                    for g in opt.param_groups:
                        g["lr"] = 3e-4 * factor
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(world, ego)
                    loss = model.loss(out, poses, target, anchor_index.index_select(0, idx).to(device))
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    loss_sum += float(loss.detach()) * len(idx)
                record = {"epoch": epoch, "train_loss": loss_sum / n}
                if epoch % 10 == 0:
                    model.eval(); preds = []
                    with torch.inference_mode():
                        for off in range(0, len(val_tokens), 64):
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                preds.append(model(val_world[off:off + 64].to(device), norm(val_ego[off:off + 64]).to(device))["poses"].float().cpu())
                    preds = torch.cat(preds)
                    record["navtrain_headval_ade_4s_m"] = float(torch.linalg.vector_norm(preds[..., :2] - val_poses[..., :2], dim=-1).mean())
                history.append(record)
            weights = folder / f"epoch-{EPOCHS:03d}.safetensors"
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights))
            model.eval(); preds = []
            with torch.inference_mode():
                for i in range(len(dev_tokens)):                          # batch one, as deployed
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(dev_world[i:i + 1].to(device), norm(dev_ego[i:i + 1]).to(device))
                    preds.append(out["poses"].float().cpu()[0])
            preds = torch.stack(preds).numpy()
            assert np.isfinite(preds).all()
            np.savez(folder / "navtest-dev-predictions.npz", scene_tokens=np.asarray(dev_tokens), trajectories=preds)
            write(folder / "history.json", history)
            write(folder / "summary.json", {"data": a.data, "kind": kind, "phase": a.phase, "seed": seed, "train_scenes": n,
                "parameters": parameter_count(model), "epochs": EPOCHS, "steps": step, "checkpoint": str(weights),
                "checkpoint_sha256": sha(weights), "anchors_K": int(anchors.shape[0]) if kind == "A" else None,
                "elapsed_seconds": time.monotonic() - start, "protocol": "pre-registration §6 (round 3)",
                "feature_integrity": "pilot/dev shards verified by SHA-256 on 2026-10-01/02 (size re-checked); X1 shards SHA-256 verified at load",
                "completed_at_utc": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"done": folder.name, "final_headval_ade": history[-1].get("navtrain_headval_ade_4s_m")}), flush=True)
            del model, opt
            torch.cuda.empty_cache()
