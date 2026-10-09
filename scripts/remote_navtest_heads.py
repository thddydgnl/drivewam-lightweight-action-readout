"""Evaluate pilot v1 heads and train protocol-v2 heads; predict on navtest-dev.

  v1     selected v1 checkpoints (unchanged) -> navtest-dev predictions
  train  protocol v2 (pre-registered 2026-10-02): fixed 40 epochs, cosine, no
         early stopping, last checkpoint only, no selection on any eval set.
The backbone is absent. navtest-dev features come from the quality run.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from lightweight_heads import build_head, parameter_count, decode_poses, action_loss
from remote_train_heads import CONFIGS, PHASES, load_dataset, batch, evaluate, gradient_gate, seed_all, sha, write

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
EPOCHS = 40

def load_dev(run_id, need_dense):
    out = ROOT / "runs" / run_id
    tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
    rows = {}
    for receipt in sorted(out.glob("shard*/*.json")):
        report = json.loads(receipt.read_text())
        assert sha(report["features_path"]) == report["features_sha256"]
        with safe_open(report["features_path"], framework="pt", device="cpu") as stored:
            mean = stored.get_tensor("mean")[:, :, 5].contiguous()
            ego = stored.get_tensor("ego").clone()
            dense = stored.get_tensor("dense").index_select(2, torch.tensor([1, 3, 5])) if need_dense else None
        for i, token in enumerate(report["tokens"]):
            rows[token] = (mean[i], ego[i], dense[i] if need_dense else None)
    assert set(rows) == set(tokens) and len(tokens) == 3250
    data = {"mean": torch.stack([rows[t][0] for t in tokens]), "ego": torch.stack([rows[t][1] for t in tokens])}
    if need_dense:
        data["dense"] = torch.stack([rows[t][2] for t in tokens])
    return tokens, data

def predict(model, data, stats, config, device):
    phase, kind = PHASES[config["phase"]], config["kind"]
    model.eval()
    outputs = []
    with torch.inference_mode():
        # Batch one, as in the deployed online policy: the BF16-autocast output
        # layer rounds differently under other batch kernels (cm-level).
        for offset in range(0, len(data["ego"]), 1):
            index = torch.arange(offset, offset + 1)
            ego = ((data["ego"].index_select(0, index) - stats["ego_mean"]) / stats["ego_std"]).to(device)
            if kind in ["sf1", "sf1-wide", "sf3"]:
                world = data["dense"][:, phase].index_select(0, index)
                if kind != "sf3":
                    world = world[:, -1:]
            else:
                world = data["mean"][:, phase].index_select(0, index)
                if kind == "linear":
                    world = (world - stats["world_mean"][phase]) / stats["world_std"][phase]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = model(world.to(device), ego)
            poses = decode_poses(normalized).cpu()
            assert torch.isfinite(poses).all()
            outputs.append(poses)
    return torch.cat(outputs).numpy()

def v1(args, device):
    out = ROOT / "runs" / args.output_run
    out.mkdir(parents=True, exist_ok=True)
    tokens, dev = load_dev(args.quality_run, True)
    pilot = ROOT / "runs/head_pilot_2026-10-01"
    report = {}
    for name, config in CONFIGS.items():
        group = "fusion" if config["kind"] in ["sf1", "sf1-wide", "sf3"] else "probes"
        stats = load_file(str(pilot / f"{group}-stats.safetensors"))
        summary = json.loads((pilot / name / "summary.json").read_text())
        checkpoint = summary["best_by_pilot_PDMS"]["checkpoint"]
        assert sha(checkpoint) == summary["best_by_pilot_PDMS"]["checkpoint_sha256"]
        model = build_head(config["kind"])
        model.load_state_dict(load_file(checkpoint), strict=True)
        model = model.to(device)
        poses = predict(model, dev, stats, config, device)
        path = out / f"{name}.npz"
        np.savez(path, scene_tokens=np.asarray(tokens), trajectories=poses)
        report[name] = {"checkpoint": checkpoint, "epoch": summary["best_by_pilot_PDMS"]["epoch"], "predictions": str(path)}
    write(out / "v1-predictions.json", {"models": report, "split": "navtest_dev", "quality_run": args.quality_run,
          "note": "v1 checkpoints were selected on navtrain pilot1000; no navtest-dev selection"})

def train(args, device):
    out = ROOT / "runs" / args.output_run / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    names = args.models.split(",") if args.models else list(CONFIGS)
    need_dense = any(CONFIGS[n]["kind"] in ["sf1", "sf1-wide", "sf3"] for n in names)
    data, index, stats, _, _ = load_dataset(ROOT / "caches/features/world_features_pilot_2026-10-01", need_dense)
    save_file({k: v.contiguous() for k, v in stats.items()}, str(out / "stats.safetensors"))
    protocol = {"protocol": "v2", "preregistration": "reports/사전등록_navtest-dev_2026-10-02.md", "epochs": EPOCHS,
        "early_stopping": False, "checkpoint": "last epoch only", "selection_on_eval_set": False, "seed": args.seed,
        "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": .01, "batch_size": 32, "warmup_fraction": .05,
        "schedule": "cosine to zero", "gradient_clip_norm": 1., "amp": "bfloat16", "train_scenes": 4000,
        "train_features": "world_features_pilot_2026-10-01 head_train rows", "monitor": "navtrain pilot head-val ADE per epoch (not used for selection)",
        "eval": "navtest_dev 3250 scenes, last checkpoint, once", "scripts": {n: sha(ROOT / "scripts" / n) for n in
            ["remote_navtest_heads.py", "remote_train_heads.py", "lightweight_heads.py"]}}
    write(out / f"protocol-{args.models or 'all'}.json", protocol)
    train_idx, val_idx = torch.arange(4000), torch.arange(4000, 5000)
    for name in names:
        folder = out / name
        if (folder / "summary.json").exists():
            continue
        folder.mkdir(exist_ok=True)
        config = CONFIGS[name]
        seed_all(args.seed)
        model = build_head(config["kind"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, betas=(.9, .999))
        total = EPOCHS * math.ceil(4000 / 32)
        warmup, step, history, gate = int(total * .05), 0, [], None
        start = time.monotonic()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = train_idx[torch.randperm(4000, generator=torch.Generator().manual_seed(args.seed + 100003 * epoch))]
            loss_sum = 0.
            for offset in range(0, 4000, 32):
                world, ego, poses, target = batch(data, stats, config, order[offset:offset + 32], device)
                step += 1
                factor = step / max(1, warmup) if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = 3e-4 * factor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = action_loss(model(world, ego), poses, target)
                assert torch.isfinite(loss)
                loss.backward()
                if gate is None:
                    gate = gradient_gate(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(order[offset:offset + 32])
            _, metrics = evaluate(model, data, stats, config, val_idx, device)
            history.append({"epoch": epoch, "step": step, "train_loss": loss_sum / 4000, "navtrain_headval_" + "ade_4s_m": metrics["ade_4s_m"]})
        weights = folder / f"epoch-{EPOCHS:03d}.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights))
        write(folder / "history.json", history)
        write(folder / "summary.json", {"model": name, "config": config, "seed": args.seed, "parameters": parameter_count(model),
              "epochs": EPOCHS, "steps": step, "checkpoint": str(weights), "checkpoint_sha256": sha(weights),
              "gradient_gate": gate, "elapsed_seconds": time.monotonic() - start,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"model": name, "seed": args.seed, "final_headval_ade": history[-1]["navtrain_headval_ade_4s_m"]}), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

def predict_v2(args, device):
    root = ROOT / "runs" / args.output_run
    tokens, dev = load_dev(args.quality_run, True)
    report = {}
    for summary_path in sorted(root.glob("seed*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        config = summary["config"]
        dense = config["kind"] in ["sf1", "sf1-wide", "sf3"]
        stats = load_file(str(summary_path.parent.parent / "stats.safetensors"))
        assert sha(summary["checkpoint"]) == summary["checkpoint_sha256"]
        model = build_head(config["kind"])
        model.load_state_dict(load_file(summary["checkpoint"]), strict=True)
        poses = predict(model.to(device), dev, stats, config, device)
        path = summary_path.parent / "navtest-dev-predictions.npz"
        np.savez(path, scene_tokens=np.asarray(tokens), trajectories=poses)
        report[f"{summary['model']}@seed{summary['seed']}"] = str(path)
    assert len(report) == 2 * len(CONFIGS), len(report)
    write(root / "v2-predictions.json", {"predictions": report, "split": "navtest_dev", "checkpoint": "last epoch"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["v1", "train", "predict-v2"])
    parser.add_argument("--quality-run", default="navtest_dev_quality_2026-10-02")
    parser.add_argument("--output-run", default="heads_navtest_dev_2026-10-02")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--models", default="")
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    if args.stage == "v1":
        v1(args, device)
    elif args.stage == "train":
        train(args, device)
    else:
        predict_v2(args, device)
