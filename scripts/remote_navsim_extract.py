"""Safely extract verified maps and navtrain logs, then freeze scene/log splits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import random
import shutil
import tarfile
import zipfile

import numpy as np
import yaml

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
FILTER_DIR = ROOT / "upstream/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter"
train = yaml.safe_load((FILTER_DIR / "navtrain.yaml").read_text())
test = yaml.safe_load((FILTER_DIR / "navtest.yaml").read_text())
train_logs, train_tokens = set(train["log_names"]), set(train["tokens"])
assert not train_logs.intersection(test["log_names"])
assert not train_tokens.intersection(test["tokens"])

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

def verified_archive(name):
    path = ROOT / "data/archives" / name
    receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    assert path.stat().st_size == receipt["bytes"]
    return path, receipt

def copy_member(stream, dest, expected):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".partial")
    h = hashlib.sha256()
    total = 0
    with temp.open("wb") as output:
        for block in iter(lambda: stream.read(1024**2), b""):
            output.write(block)
            h.update(block)
            total += len(block)
    assert total == expected
    if dest.exists():
        assert dest.stat().st_size == total
        with dest.open("rb") as old:
            old_hash = hashlib.sha256()
            for block in iter(lambda: old.read(1024**2), b""):
                old_hash.update(block)
        assert old_hash.hexdigest() == h.hexdigest(), f"Conflicting existing file: {dest}"
        temp.unlink()
    else:
        temp.replace(dest)
    return {"path": str(dest), "bytes": total, "sha256": h.hexdigest()}

def maps():
    path, receipt = verified_archive("nuplan-maps-v1.1.zip")
    records = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            assert parts and ".." not in parts and not member.filename.startswith("/")
            assert parts[0] == "nuplan-maps-v1.0" or parts == ("LICENSE",)
            assert ((member.external_attr >> 16) & 0o170000) != 0o120000
            if member.is_dir():
                continue
            dest = ROOT / "data/maps" / (Path(*parts[1:]) if parts[0] == "nuplan-maps-v1.0" else Path("LICENSE"))
            with archive.open(member) as stream:
                records.append(copy_member(stream, dest, member.file_size))
    assert (ROOT / "data/maps/nuplan-maps-v1.0.json").exists()
    write_json(ROOT / "reports/navsim-maps-extraction.json", {
        "archive": receipt, "map_version": "nuplan-maps-v1.0", "files": records,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print("maps_ready", len(records), sum(r["bytes"] for r in records), flush=True)

def metadata():
    path, receipt = verified_archive("openscene_metadata_trainval.tgz")
    records = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            assert not member.name.startswith("/") and ".." not in parts
            if not member.isfile():
                assert member.isdir(), f"Unexpected archive link: {member.name}"
                continue
            if Path(parts[-1]).stem not in train_logs or not parts[-1].endswith(".pkl"):
                continue
            dest = ROOT / "data/navsim_logs/trainval" / parts[-1]
            with archive.extractfile(member) as stream:
                records.append(copy_member(stream, dest, member.size))
            if len(records) % 100 == 0:
                write_json(ROOT / "reports/navsim-metadata-status.json", {"stage": "extracting", "logs": len(records)})
                print("extracted_logs", len(records), flush=True)
    extracted = {Path(r["path"]).stem for r in records}
    assert extracted == train_logs, f"Missing navtrain logs: {train_logs - extracted}"
    write_json(ROOT / "reports/navsim-metadata-extraction.json", {
        "archive": receipt, "scope": "official navtrain logs only", "files": records,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    index()

def index():
    ordered_logs = sorted(train_logs)
    random.Random(17).shuffle(ordered_logs)
    n_val = math.ceil(len(ordered_logs) * 0.1)
    val_logs = set(ordered_logs[:n_val])
    split = {"seed": 17, "algorithm": "Python random.Random(17).shuffle(sorted(log_names)); first ceil(10%) to head-val",
             "head_train": sorted(train_logs - val_logs), "head_val": sorted(val_logs),
             "navtrain_yaml_sha256": hashlib.sha256((FILTER_DIR / "navtrain.yaml").read_bytes()).hexdigest(),
             "navtest_yaml_sha256": hashlib.sha256((FILTER_DIR / "navtest.yaml").read_bytes()).hexdigest(),
             "navtest_locked": True}
    write_json(ROOT / "splits/navtrain-log-split-seed17.json", split)
    seen = set()
    counts = Counter()
    per_log = {}
    index_path = ROOT / "splits/navtrain-scene-index.jsonl"
    with index_path.with_suffix(".partial").open("w") as output:
        for number, log in enumerate(sorted(train_logs)):
            frames = pickle.load((ROOT / "data/navsim_logs/trainval" / (log + ".pkl")).open("rb"))
            log_count = 0
            for start in range(0, len(frames), train["frame_interval"]):
                if start + 14 > len(frames):
                    continue
                current = frames[start + 3]
                token = current["token"]
                if token not in train_tokens or (train["has_route"] and not current["roadblock_ids"]):
                    continue
                assert token not in seen
                assert current["log_name"] == log
                seen.add(token)
                command = np.asarray(current["driving_command"])
                image = PurePosixPath(current["cams"]["CAM_F0"]["data_path"])
                assert not image.is_absolute() and ".." not in image.parts
                row = {"scene_token": token, "log_name": log,
                       "split": "head_val" if log in val_logs else "head_train",
                       "image_path": str(ROOT / "data/sensor_blobs/trainval" / str(image)),
                       "map_name": current["map_location"], "command_id": int(command.argmax()),
                       "timestamp_us": int(current["timestamp"]), "frame_start_index": start,
                       "label_horizon_s": 4.0, "metadata_future_horizon_s": 5.0,
                       "label_path": str(ROOT / "processed/navtrain-current-v1/targets" / (token + ".npz")),
                       "input_available": False, "target_available": False, "metric_cache_available": False,
                       "speed_mps": float(np.linalg.norm(np.asarray(current["ego_dynamic_state"][:2])))}
                output.write(json.dumps(row) + "\n")
                counts[row["split"]] += 1
                log_count += 1
            per_log[log] = log_count
            del frames
            if (number + 1) % 100 == 0:
                write_json(ROOT / "reports/navsim-metadata-status.json", {"stage": "indexing", "logs": number + 1, "scenes": len(seen)})
                print("indexed_logs", number + 1, "scenes", len(seen), flush=True)
    assert seen == train_tokens, f"Token coverage mismatch missing={len(train_tokens-seen)} extra={len(seen-train_tokens)}"
    index_path.with_suffix(".partial").replace(index_path)
    write_json(ROOT / "reports/navsim-scene-index.json", {
        "logs": len(train_logs), "scenes": len(seen), "split_counts": dict(counts),
        "per_log_scene_counts": per_log, "navtest_token_overlap": 0, "navtest_log_overlap": 0,
        "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(ROOT / "reports/navsim-metadata-status.json", {"stage": "verified", "logs": len(train_logs), "scenes": len(seen)})
    print("index_ready", dict(counts), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["maps", "metadata", "index"])
    args = parser.parse_args()
    globals()[args.stage]()
