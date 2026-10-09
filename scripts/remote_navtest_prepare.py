"""Prepare pre-registered navtest-dev (and later navtest-final) observation inputs.

Stages, all in the scorer environment:
  metadata  extract the 136 navtest log pickles from the verified metadata archive
  index     index all 12,146 navtest tokens and assign navtest_dev/final by frozen log split
  extract   stream verified camera archives, keep only current-frame CAM_F0 images
  process   write observation-only inputs and separate 4 s targets for one split
Inputs never contain future pixels or labels. navtest-final is processed only
when explicitly requested at the locked final-evaluation stage.
"""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import tarfile
import time

import numpy as np
import yaml

from remote_navsim_extract import copy_member, write_json, verified_archive

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
FILTER = ROOT / "upstream/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
LOGS = ROOT / "data/navsim_logs/test"
BLOBS = ROOT / "data/sensor_blobs/test"
OUT = ROOT / "processed/navtest-current-v1"
SPLITS = ROOT / "splits"
STATUS = ROOT / "reports/navtest-prepare-status.json"
SPLIT_SHA = "68178d2266df41c3f5d618f81ba75f00a9b99589bab026f271a8b82b4ef031a4"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def frozen_split():
    path = SPLITS / "navtest-log-split-seed17.json"
    assert sha(path) == SPLIT_SHA, "Pre-registered navtest split changed"
    return json.loads(path.read_text())

def metadata():
    navtest = yaml.safe_load(FILTER.read_text())
    logs = set(navtest["log_names"])
    path, receipt = verified_archive("openscene_metadata_test.tgz")
    assert receipt["publisher_sha256_verified"]
    records = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            assert not member.name.startswith("/") and ".." not in parts
            if not member.isfile():
                assert member.isdir(), f"Unexpected archive link: {member.name}"
                continue
            if not parts[-1].endswith(".pkl") or Path(parts[-1]).stem not in logs:
                continue
            with archive.extractfile(member) as stream:
                records.append(copy_member(stream, LOGS / parts[-1], member.size))
    assert {Path(r["path"]).stem for r in records} == logs, "Missing navtest logs"
    write_json(ROOT / "reports/navtest-metadata-extraction.json", {"archive": receipt, "logs": len(records),
               "files": records, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print("navtest_logs", len(records), flush=True)

def index():
    navtest = yaml.safe_load(FILTER.read_text())
    tokens, split = set(navtest["tokens"]), frozen_split()
    dev_logs, final_logs = set(split["navtest_dev_logs"]), set(split["navtest_final_logs"])
    assert dev_logs | final_logs == set(navtest["log_names"]) and not dev_logs & final_logs
    rows, seen = [], set()
    for log in sorted(navtest["log_names"]):
        frames = pickle.load((LOGS / (log + ".pkl")).open("rb"))
        for start in range(0, len(frames), navtest["frame_interval"]):
            if start + 14 > len(frames):
                continue
            current = frames[start + 3]
            token = current["token"]
            if token not in tokens or (navtest["has_route"] and not current["roadblock_ids"]):
                continue
            assert token not in seen and current["log_name"] == log
            seen.add(token)
            image = PurePosixPath(current["cams"]["CAM_F0"]["data_path"])
            assert not image.is_absolute() and ".." not in image.parts
            name = "navtest_dev" if log in dev_logs else "navtest_final"
            rows.append({"scene_token": token, "log_name": log, "split": name,
                "image_relative": str(image), "image_path": str(BLOBS / str(image)),
                "map_name": current["map_location"], "command_id": int(np.asarray(current["driving_command"]).argmax()),
                "timestamp_us": int(current["timestamp"]), "frame_start_index": start,
                "speed_mps": float(np.linalg.norm(np.asarray(current["ego_dynamic_state"][:2]))),
                "input_path": str(OUT / "inputs" / (token + ".npz")), "label_path": str(OUT / "targets" / (token + ".npz")),
                "input_available": False, "target_available": False})
        del frames
    assert seen == tokens, f"navtest coverage mismatch missing={len(tokens - seen)}"
    (SPLITS / "navtest-scene-index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dev = [r for r in rows if r["split"] == "navtest_dev"]
    (SPLITS / "navtest-dev-tokens.txt").write_text("".join(r["scene_token"] + "\n" for r in dev))
    config = {"_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all",
              "num_history_frames": 4, "num_future_frames": 10, "frame_interval": 1, "has_route": True,
              "max_scenes": None, "log_names": sorted({r["log_name"] for r in dev}), "tokens": [r["scene_token"] for r in dev]}
    (SPLITS / "navtest-dev.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = {}
    for name in ["navtest_dev", "navtest_final"]:
        subset = [r for r in rows if r["split"] == name]
        report[name] = {"scenes": len(subset), "logs": len({r["log_name"] for r in subset}),
            "maps": dict(Counter(r["map_name"] for r in subset)), "commands": dict(Counter(r["command_id"] for r in subset)),
            "speed_mps_quantiles": np.quantile([r["speed_mps"] for r in subset], [0, .25, .5, .75, 1]).tolist(),
            "largest_log_fraction": max(Counter(r["log_name"] for r in subset).values()) / len(subset)}
    write_json(ROOT / "reports/navtest-index.json", {"split_sha256": SPLIT_SHA, **report,
        "index_sha256": sha(SPLITS / "navtest-scene-index.jsonl"), "dev_tokens_sha256": sha(SPLITS / "navtest-dev-tokens.txt"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(report), flush=True)

def extract():
    rows = [json.loads(l) for l in (SPLITS / "navtest-scene-index.jsonl").read_text().splitlines()]
    needed = {r["image_relative"] for r in rows}
    specs = [s for s in json.loads((ROOT / "reports/navtest-download-specs.json").read_text()) if "camera" in s["name"]]
    receipts = ROOT / "reports/navtest-camera-extraction"
    receipts.mkdir(exist_ok=True)

    def one(spec):
        name = spec["name"]
        path = ROOT / "data/archives" / name
        deadline = time.monotonic() + 4 * 3600
        while not path.with_suffix(path.suffix + ".json").exists():
            assert time.monotonic() < deadline, f"Timed out waiting for {name}"
            time.sleep(10)
        path, receipt = verified_archive(name)
        assert receipt["publisher_sha256_verified"]
        report_path = receipts / (name + ".json")
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if all(Path(r["path"]).is_file() and Path(r["path"]).stat().st_size == r["bytes"] for r in prior["files"]):
                return prior
        records = []
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                assert not member.name.startswith("/") and ".." not in parts
                if not member.isfile():
                    assert member.isdir(), f"Unexpected archive link: {member.name}"
                    continue
                if "CAM_F0" not in parts:
                    continue
                relative = str(PurePosixPath(*parts[parts.index("CAM_F0") - 1:]))
                if relative in needed:
                    with archive.extractfile(member) as stream:
                        records.append(copy_member(stream, BLOBS / relative, member.size))
        result = {"archive": receipt, "files": records, "only_current_front_images": True,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(report_path, result)
        print("extracted", name, len(records), flush=True)
        return result

    write_json(STATUS, {"stage": "extracting_front_images", "archives": len(specs), "required_images": len(needed)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one, specs))
    missing = [p for p in needed if not (BLOBS / p).is_file()]
    assert not missing, f"Missing images: {len(missing)} {missing[:5]}"
    write_json(ROOT / "reports/navtest-camera-extraction.json", {"archives": len(results), "front_images": len(needed),
        "extracted_bytes": sum(r["bytes"] for res in results for r in res["files"]), "missing": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(STATUS, {"stage": "front_images_verified", "images": len(needed)})

def process(split):
    import cv2
    from PIL import Image
    from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
    cv2.setNumThreads(1)
    assert split in ["navtest_dev", "navtest_final"]
    rows = [r for r in (json.loads(l) for l in (SPLITS / "navtest-scene-index.jsonl").read_text().splitlines()) if r["split"] == split]
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "targets").mkdir(parents=True, exist_ok=True)
    per_log = defaultdict(list)
    for row in rows:
        per_log[row["log_name"]].append(row)
    no_sensors = SensorConfig.build_no_sensors()

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
            future = scene.get_future_trajectory().poses.astype(np.float32)
            assert future.shape == (10, 3)
            deltas = np.diff(np.concatenate([np.zeros((1, 3), np.float32), future[:8]]), axis=0)
            np.testing.assert_allclose(np.cumsum(deltas, axis=0), future[:8], rtol=0, atol=1e-5)
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
            assert all(np.isfinite(v).all() for v in values.values()) and np.isfinite(future).all()
            np.savez(row["input_path"], **values)
            np.savez(row["label_path"], future_poses_5s=future, future_poses_4s=future[:8], action_deltas_4s=deltas)
            row.update(input_available=True, target_available=True,
                       input_sha256=sha(row["input_path"]), target_sha256=sha(row["label_path"]))
        return len(samples)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for count in pool.map(one, sorted(per_log.items())):
            done += count
            write_json(STATUS, {"stage": "processing", "split": split, "completed": done, "total": len(rows)})
    assert done == len(rows)
    manifest = SPLITS / f"{split.replace('_', '-')}-manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(ROOT / f"reports/{split.replace('_', '-')}-processing.json", {"split": split, "samples": done,
        "manifest_sha256": sha(manifest), "input_contains_future_pixels_or_labels": False,
        "canonical_history_parity_checked": done, "trajectory_cumsum_roundtrip_checked": done,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(STATUS, {"stage": "processed_and_verified", "split": split, "samples": done})
    print("processed", split, done, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["metadata", "index", "extract", "process"])
    parser.add_argument("--split", default="navtest_dev")
    args = parser.parse_args()
    try:
        if args.stage == "process":
            process(args.split)
        else:
            globals()[args.stage]()
    except Exception as error:
        write_json(STATUS, {"stage": "failed", "operation": args.stage, "error": repr(error)})
        raise
