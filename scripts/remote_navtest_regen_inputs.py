"""Regenerate deleted navtest observation inputs (pre-registration 2026-11-01 §4).

The 2026-10-05 cleanup removed processed/navtest-current-v1/inputs. This rewrites them from the same
logs and sensor blobs with the input half of `process` in remote_navtest_prepare.py. It never opens
or rewrites targets and never touches the frozen manifests: npz zip timestamps make the bytes (not
the arrays) differ from the originals, so the new input hashes go to
splits/<split>-manifest.regen-2026-11-01[-<tag>].jsonl. Content is verified downstream, where
B02-opt and the student must reproduce stored trajectories. Scorer environment, CPU only.
"""
import argparse
from collections import defaultdict
import concurrent.futures
from datetime import datetime, timezone
import json
import pickle

import numpy as np

from remote_navtest_prepare import ROOT, LOGS, BLOBS, SPLITS, sha, write_json

def regenerate(split, tokens, tag, keep_from, workers):
    import cv2
    from PIL import Image
    from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
    cv2.setNumThreads(1)
    name = split.replace("_", "-")
    status = ROOT / f"reports/r5-regen-{name}{'-' + tag if tag else ''}.json"
    rows = [json.loads(line) for line in (SPLITS / f"{name}-manifest.jsonl").read_text().splitlines()]
    if tokens is not None:
        wanted = set(tokens)
        rows = [r for r in rows if r["scene_token"] in wanted]
        assert len(rows) == len(wanted), "tokens missing from the manifest"
    kept = {}
    for path in keep_from:                                     # already regenerated: keep bytes and hash
        for r in map(json.loads, path.read_text().splitlines()):
            assert sha(r["input_path"]) == r["input_sha256"], r["scene_token"]
            kept[r["scene_token"]] = r
    (ROOT / "processed/navtest-current-v1/inputs").mkdir(parents=True, exist_ok=True)
    per_log = defaultdict(list)
    for row in rows:
        if row["scene_token"] not in kept:
            per_log[row["log_name"]].append(row)
    no_sensors = SensorConfig.build_no_sensors()
    new = {}

    def one(item):
        log, samples = item
        frames = pickle.load((LOGS / (log + ".pkl")).open("rb"))
        for row in samples:
            window = frames[row["frame_start_index"]:row["frame_start_index"] + 14]
            assert window[3]["token"] == row["scene_token"]
            scene = Scene.from_scene_dict_list(window, BLOBS, 4, 10, no_sensors)
            agent = AgentInput.from_scene_dict_list(window[:4], BLOBS, 4, no_sensors)
            history = np.stack([e.ego_pose for e in agent.ego_statuses]).astype(np.float32)
            np.testing.assert_allclose(history, scene.get_history_trajectory().poses.astype(np.float32), rtol=0, atol=1e-6)
            with Image.open(row["image_path"]) as image:
                assert image.mode == "RGB"
                rgb = np.asarray(image)
            h, w = rgb.shape[:2]
            ego = agent.ego_statuses[-1]
            values = {"rgb_current": cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                      "history_poses": history,
                      "ego_velocity_acc": np.concatenate([ego.ego_velocity, ego.ego_acceleration]).astype(np.float32),
                      "command_id": np.asarray(int(np.argmax(ego.driving_command)), dtype=np.int64)}
            assert int(values["command_id"]) == row["command_id"]
            assert all(np.isfinite(v).all() for v in values.values())
            np.savez(row["input_path"], **values)
            new[row["scene_token"]] = {**row, "input_sha256": sha(row["input_path"]),
                                       "input_sha256_2026_10": row["input_sha256"]}
        return len(samples)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for count in pool.map(one, sorted(per_log.items())):
            done += count
            write_json(status, {"stage": "regenerating", "split": split, "completed": done + len(kept), "total": len(rows)})
    out_rows = [kept.get(r["scene_token"]) or new[r["scene_token"]] for r in rows]
    manifest = SPLITS / f"{name}-manifest.regen-2026-11-01{'-' + tag if tag else ''}.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    write_json(status, {"stage": "verified", "split": split, "samples": len(out_rows), "regenerated": done,
                        "kept": len(kept), "manifest": str(manifest), "manifest_sha256": sha(manifest),
                        "targets_opened": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print("regenerated", split, tag, done, "kept", len(kept), flush=True)

if __name__ == "__main__":
    import hashlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["navtest_dev", "navtest_final"])
    parser.add_argument("--latency256", action="store_true", help="only the 256 fixed final latency scenes")
    parser.add_argument("--keep-from", default="", help="comma list of regen manifests whose inputs are kept")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    tokens, tag = None, None
    if args.latency256:
        assert args.split == "navtest_final"
        rows = [json.loads(line) for line in (SPLITS / "navtest-final-manifest.jsonl").read_text().splitlines()]
        tokens = sorted((r["scene_token"] for r in rows),
                        key=lambda t: hashlib.sha256(f"17:final-latency:{t}".encode()).hexdigest())[:256]
        tag = "latency256"
    regenerate(args.split, tokens, tag, [ROOT / p for p in args.keep_from.split(",") if p], args.workers)
