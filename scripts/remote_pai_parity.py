"""PhysicalAI parity gates (must pass before any baseline/student number is used).

1. data: upstream PhysicalAITestDataset on the kept 10 Hz mp4 == ClipData on frames_1fps.npz
2. rollout: upstream predict_autoregressive == pai_common.rollout for B10 and B05 (bitwise)
3. opt: skipping the terminal action pass leaves B10/B05 outputs bitwise unchanged
4. fused: rollout_multi (shared video passes) == separate runs for every baseline, bitwise
5. taps: hooks on/off do not change B10 outputs; early-exit student features == B10-hooked features
`--set final --n 3` is the gate (needs >= 2 processed pool clips for check 1); `--set dev --n 32`
is the pre-registered 32-clip record on dev. Model environment, one GPU.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
import pai_common as pc                                          # noqa: E402

def h(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

SPECS = [("B10-opt", 10, True), ("B05-opt", 5, True), ("B03-opt", 3, True), ("B02-opt", 2, True),
         ("B01-opt", 1, True), ("B10", 10, False)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="final"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--taps-only", action="store_true", help="record check 5 only (hooked B10 vs early exit)")
    args = ap.parse_args()
    assert torch.cuda.device_count() == 1
    report = {"utc": datetime.now(timezone.utc).isoformat(), "set": args.set, "data": [], "rollout": [], "opt": [],
              "fused": [], "taps": []}
    cfg = pc.config()
    if args.set == "final":
        store = json.loads((ROOT / "work/DriveWAM/src/physicalai/eval_data/prompts_test_sample_1k.json").read_text())
    else:
        store = json.loads((ROOT / "data/physicalai/dev_guidance/scene_evolving_guidance_store.json").read_text())
    # 1. data parity on pool clips that kept the 10 Hz mp4
    pool = [c for c in (ROOT / "splits/pai-devpool-3k.txt").read_text().split() if (ROOT / "data/physicalai/front" / c / "camera_front_wide_120fov.mp4").exists()][:4]
    assert len(pool) >= 2, "need processed pool clips with mp4"
    up_cfg = pc.config(); up_cfg.dataset_path = str(ROOT / "data/physicalai/front")
    upstream = pc.PhysicalAITestDataset(up_cfg, max_clips=len(pool), clip_ids=pool, clip_prompts_path=None)
    ours = pc.ClipData(cfg, pool)
    for i, c in enumerate(pool):
        a, b = upstream[upstream.clip_ids.index(c)], ours[i]
        same = all(torch.equal(a[k], b[k]) for k in ["images", "actions", "actions_mask", "ego_velocity_acc", "curvature"]) and a["prompts"] == b["prompts"]
        report["data"].append({"clip": c, "equal": bool(same)}); assert same, c
    # 2-5. rollout parity (published prompts for final, generated guidance for dev)
    cache = pc.PromptCache(sorted((ROOT / "caches/pai-prompts" / args.set).glob("*.safetensors")))
    ev = pc.load_policy("cuda:0", cache)
    clips = (ROOT / f"splits/pai-{args.set}-1k.txt").read_text().split()[:args.n]
    data = pc.ClipData(cfg, clips, prompts_store=store)
    for i, c in enumerate(clips):
        batch = pc.collate(data[i])
        if args.taps_only:
            up_a10, up_l10 = ev.predict_autoregressive(batch, num_video_steps=3, num_action_steps=10, seed=0,
                                                       video_denoise_fraction=0.6)
        separate = {}
        for steps in ([] if args.taps_only else [10, 5]):
            up_a, up_l = ev.predict_autoregressive(batch, num_video_steps=3, num_action_steps=steps, seed=0, video_denoise_fraction=0.6)
            my_a, my_l, _, calls = pc.rollout(ev, batch, "B", 3, steps)
            op_a, op_l, _, op_calls = pc.rollout(ev, batch, "B", 3, steps, skip_terminal_action=True)
            ok = h(up_a) == h(my_a) and h(up_l) == h(my_l)
            ok_opt = h(op_a) == h(up_a) and h(op_l) == h(up_l)
            report["rollout"].append({"clip": c, "steps": steps, "equal": ok, "calls": calls})
            report["opt"].append({"clip": c, "steps": steps, "equal": ok_opt, "calls": op_calls})
            assert ok and ok_opt, (c, steps, ok, ok_opt)
            separate[f"B{steps:02d}-opt"] = h(op_a)
            if steps == 10:
                separate["B10"], up_a10, up_l10 = h(my_a), up_a, up_l
        for name, steps, skip in ([] if args.taps_only else SPECS):
            if name not in separate:
                separate[name] = h(pc.rollout(ev, batch, "B", 3, steps, skip_terminal_action=skip)[0])
        if not args.taps_only:
            fused, fused_l, fused_calls = pc.rollout_multi(ev, batch, SPECS)
            ok_fused = all(h(fused[n]) == separate[n] for n, _, _ in SPECS) and h(fused_l) == h(up_l10)
            report["fused"].append({"clip": c, "equal": ok_fused, "calls": fused_calls})
            assert ok_fused, c
        up_a, up_l = up_a10, up_l10
        tap = pc.Tap(ev)
        hk_a, hk_l, feats_b, _ = pc.rollout(ev, batch, "B", 3, 10, tap=tap, collect=("current", "first", "terminal"))
        st_a, _, feats_s, s_calls = pc.rollout(ev, batch, "S", 3, 10, tap=tap, collect=("current", "first"), stop_video_after_first=True)
        tap.close()
        same_out = h(hk_a) == h(up_a) and h(hk_l) == h(up_l)
        same_feat = all(torch.equal(fb["first"], fs["first"]) and torch.equal(fb["current"], fs["current"])
                        for fb, fs in zip(feats_b, feats_s))
        shapes = [tuple(f["first"].shape) for f in feats_b]
        report["taps"].append({"clip": c, "hooked_outputs_equal": same_out, "early_exit_features_equal": same_feat,
                               "student_calls": s_calls, "feature_shapes": shapes})
        assert same_out and same_feat, c
    report["passed"] = True
    name = "pai-parity.json" if args.set == "final" else f"pai-parity-{args.set}{args.n}.json"
    (ROOT / "reports" / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "clips": len(clips), "student_calls": report["taps"][0]["student_calls"],
                      "shapes": report["taps"][0]["feature_shapes"]}))
