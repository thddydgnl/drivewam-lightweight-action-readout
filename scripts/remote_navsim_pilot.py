"""Freeze a 4k/1k pilot and prepare observation-only inputs / separate GT.

Use the scorer environment for canonical NAVSIM geometry and the Haswell BLAS
workaround. No policy training or navtest evaluation is performed here.
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

import cv2
import numpy as np
from PIL import Image
import yaml

from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from remote_navsim_extract import copy_member, write_json, verified_archive

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
OUT = ROOT / "processed/navtrain-current-v1"
SPLITS = ROOT / "splits"
STATUS = ROOT / "reports/navsim-pilot-status.json"
cv2.setNumThreads(1)

def rank(token, purpose):
    return hashlib.sha256(f"17:{purpose}:{token}".encode()).hexdigest()

def load_rows():
    return [json.loads(line) for line in (SPLITS / "pilot-manifest.jsonl").read_text().splitlines()]

def select():
    cohort = json.loads((ROOT / "reports/navsim-pilot-archive-selection.json").read_text())
    logs = set(cohort["head_train_logs"] + cohort["head_val_logs"])
    candidates = [json.loads(line) for line in (SPLITS / "navtrain-scene-index.jsonl").read_text().splitlines()]
    candidates = [row for row in candidates if row["log_name"] in logs]
    selected = []
    groups = {}
    for split, count in [("head_train", 4000), ("head_val", 1000)]:
        rows = sorted((r for r in candidates if r["split"] == split), key=lambda r: rank(r["scene_token"], "pilot"))[:count]
        assert len(rows) == count
        groups[split] = rows
        for row in rows:
            row["input_path"] = str(OUT / "inputs" / (row["scene_token"] + ".npz"))
            row["label_path"] = str(OUT / "targets" / (row["scene_token"] + ".npz"))
        selected.extend(rows)
    train_tokens = {r["scene_token"] for r in groups["head_train"]}
    val_tokens = {r["scene_token"] for r in groups["head_val"]}
    assert not train_tokens & val_tokens
    assert not {r["log_name"] for r in groups["head_train"]} & {r["log_name"] for r in groups["head_val"]}
    smoke = sorted(groups["head_train"], key=lambda r: rank(r["scene_token"], "smoke"))[:32]
    dev = sorted(groups["head_val"], key=lambda r: rank(r["scene_token"], "dev"))[:256]
    for name, rows in [("pilot-head-train", groups["head_train"]), ("pilot-head-val", groups["head_val"]),
                       ("smoke32", smoke), ("devval256", dev)]:
        (SPLITS / (name + "-tokens.txt")).write_text("\n".join(r["scene_token"] for r in rows) + "\n")
        config = {"_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all",
                  "num_history_frames": 4, "num_future_frames": 10, "frame_interval": 1,
                  "has_route": True, "max_scenes": None,
                  "log_names": sorted({r["log_name"] for r in rows}), "tokens": [r["scene_token"] for r in rows]}
        (SPLITS / (name + ".yaml")).write_text(yaml.safe_dump(config, sort_keys=False))
    (SPLITS / "pilot-manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    needed_images = {str(Path(r["image_path"]).relative_to(ROOT / "data/sensor_blobs/trainval")) for r in selected}
    # A small compatibility fixture may read future RGB; those pixels are not
    # included in any observation-only input or policy inference batch.
    by_log = defaultdict(list)
    for row in smoke:
        by_log[row["log_name"]].append(row)
    for log, rows in by_log.items():
        with (ROOT / "data/navsim_logs/trainval" / (log + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        for row in rows:
            start = row["frame_start_index"]
            needed_images.update(frame["cams"]["CAM_F0"]["data_path"] for frame in frames[start:start + 14])
    write_json(ROOT / "reports/navsim-pilot-required-images.json", sorted(needed_images))
    write_json(ROOT / "reports/navsim-pilot-selection.json", {
        "seed": 17, "selection": "lowest SHA-256 rank of 17:pilot:scene_token inside fixed archive cohort",
        "head_train": 4000, "head_val": 1000, "smoke32": [r["scene_token"] for r in smoke],
        "devval256": [r["scene_token"] for r in dev], "required_front_images": len(needed_images),
        "future_rgb_compatibility_only_scenes": len(smoke), "navtest_locked": True,
        "scope": "pilot; full metadata indexed, remaining full train camera data not downloaded",
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(STATUS, {"stage": "selected", "head_train": 4000, "head_val": 1000,
                        "required_front_images": len(needed_images)})
    print("pilot_selected", len(selected), "front_images", len(needed_images), flush=True)

def extract():
    needed = set(json.loads((ROOT / "reports/navsim-pilot-required-images.json").read_text()))
    specs = json.loads((ROOT / "reports/navsim-camera-download-specs.json").read_text())
    destination = ROOT / "data/sensor_blobs/trainval"
    receipts = ROOT / "reports/navsim-camera-extraction"
    receipts.mkdir(exist_ok=True)

    def extract_archive(spec):
        name = spec["name"]
        path = ROOT / "data/archives" / name
        deadline = time.monotonic() + 5400
        while not path.with_suffix(path.suffix + ".json").exists():
            assert time.monotonic() < deadline, f"Timed out waiting for {name}"
            time.sleep(5)
        path, receipt = verified_archive(name)
        assert receipt["publisher_sha256_verified"]
        records = []
        report_path = receipts / (name + ".json")
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if all(Path(r["path"]).exists() and Path(r["path"]).stat().st_size == r["bytes"] for r in prior["files"]):
                return prior
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                assert not member.name.startswith("/") and ".." not in parts
                if not member.isfile():
                    assert member.isdir(), f"Unexpected archive link: {member.name}"
                    continue
                if "CAM_F0" not in parts:
                    continue
                camera_index = parts.index("CAM_F0")
                assert camera_index >= 1
                relative = str(PurePosixPath(*parts[camera_index - 1:]))
                if relative not in needed:
                    continue
                with archive.extractfile(member) as stream:
                    records.append(copy_member(stream, destination / relative, member.size))
        result = {"archive": receipt, "files": records, "only_required_front_images": True,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(report_path, result)
        print("camera_archive_extracted", name, len(records), flush=True)
        return result

    write_json(STATUS, {"stage": "extracting_front_images", "archives": len(specs), "required_images": len(needed)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(extract_archive, specs))
    missing = [path for path in needed if not (destination / path).is_file()]
    assert not missing, f"Missing required images: {len(missing)} {missing[:10]}"
    write_json(ROOT / "reports/navsim-camera-extraction.json", {
        "archives": len(results), "front_images": len(needed),
        "extracted_bytes": sum(r["bytes"] for result in results for r in result["files"]),
        "missing_required_images": 0, "lidar_extracted": False, "other_cameras_extracted": False,
        "archive_receipts": [str(receipts / (spec["name"] + ".json")) for spec in specs],
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(STATUS, {"stage": "front_images_verified", "images": len(needed), "missing": 0})

def process():
    rows = load_rows()
    assert len(rows) == 5000
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "targets").mkdir(parents=True, exist_ok=True)
    per_log = defaultdict(list)
    for row in rows:
        per_log[row["log_name"]].append(row)
    canonical_no_sensors = SensorConfig.build_no_sensors()
    expected_input_keys = {"rgb_current", "history_poses", "ego_velocity_acc", "command_id"}

    def process_log(item):
        log, samples = item
        with (ROOT / "data/navsim_logs/trainval" / (log + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        for row in samples:
            start = row["frame_start_index"]
            frame_window = frames[start:start + 14]
            current = frame_window[3]
            assert current["token"] == row["scene_token"]
            scene = Scene.from_scene_dict_list(frame_window, ROOT / "data/sensor_blobs/trainval", 4, 10, canonical_no_sensors)
            agent = AgentInput.from_scene_dict_list(frame_window[:4], ROOT / "data/sensor_blobs/trainval", 4, canonical_no_sensors)
            history = np.stack([ego.ego_pose for ego in agent.ego_statuses]).astype(np.float32)
            canonical_history = scene.get_history_trajectory().poses.astype(np.float32)
            np.testing.assert_allclose(history, canonical_history, rtol=0, atol=1e-6)
            future = scene.get_future_trajectory().poses.astype(np.float32)
            assert future.shape == (10, 3)
            deltas = np.diff(np.concatenate([np.zeros((1, 3), dtype=np.float32), future[:8]], axis=0), axis=0)
            np.testing.assert_allclose(np.cumsum(deltas, axis=0), future[:8], rtol=0, atol=1e-5)
            with Image.open(row["image_path"]) as image:
                assert image.mode == "RGB"
                rgb = np.asarray(image)
            h, w = rgb.shape[:2]
            half = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            ego = agent.ego_statuses[-1]
            values = {"rgb_current": half, "history_poses": history,
                      "ego_velocity_acc": np.concatenate([ego.ego_velocity, ego.ego_acceleration]).astype(np.float32),
                      "command_id": np.asarray(int(np.argmax(ego.driving_command)), dtype=np.int64)}
            assert set(values) == expected_input_keys
            assert values["ego_velocity_acc"].shape == (4,) and history.shape == (4, 3)
            assert all(np.isfinite(v).all() for v in values.values()) and np.isfinite(future).all()
            np.savez(row["input_path"], **values)
            np.savez(row["label_path"], future_poses_5s=future, future_poses_4s=future[:8], action_deltas_4s=deltas)
            row.update(input_available=True, target_available=True,
                       input_sha256=hashlib.sha256(Path(row["input_path"]).read_bytes()).hexdigest(),
                       target_sha256=hashlib.sha256(Path(row["label_path"]).read_bytes()).hexdigest())
        return len(samples)

    completed = 0
    write_json(STATUS, {"stage": "processing", "completed": 0, "total": len(rows)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_log, item) for item in sorted(per_log.items())]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            write_json(STATUS, {"stage": "processing", "completed": completed, "total": len(rows)})
            print("processed", completed, flush=True)
    assert completed == len(rows) and all(r["input_available"] and r["target_available"] for r in rows)
    manifest = SPLITS / "pilot-manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    distributions = {}
    for split in ["head_train", "head_val"]:
        selected = [r for r in rows if r["split"] == split]
        distributions[split] = {"scenes": len(selected), "logs": len({r["log_name"] for r in selected}),
             "maps": dict(Counter(r["map_name"] for r in selected)),
             "commands": dict(Counter(r["command_id"] for r in selected)),
             "speed_mps_quantiles": np.quantile([r["speed_mps"] for r in selected], [0, .25, .5, .75, 1]).tolist()}
    write_json(ROOT / "reports/navsim-pilot-processing.json", {
        "samples": completed, "distributions": distributions, "input_keys": sorted(expected_input_keys),
        "input_contains_future_pixels_or_labels": False,
        "ground_truth_separate": True, "future_label_horizon_s": 4.0,
        "rgb_preprocessing": "PIL RGB uint8; cv2 INTER_AREA to half source dimensions; model subsequently uses its original bilinear 256x448 resize",
        "canonical_history_parity_checked": completed, "trajectory_cumsum_roundtrip_checked": completed,
        "all_values_finite": True, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(STATUS, {"stage": "processed_and_verified", "samples": completed, "head_train": 4000, "head_val": 1000})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "extract", "process"])
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception as error:
        write_json(STATUS, {"stage": "failed", "operation": args.stage, "error": repr(error)})
        raise
