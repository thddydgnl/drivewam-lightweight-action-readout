"""Round-4: stream every remaining navtrain head-train scene (pre-registration §7).

  select    remaining head-train tokens (not pilot-train, not X1) and the archives holding them
  run       per archive: download (verified) -> extract current CAM_F0 -> process inputs/targets
            -> write fragment -> delete the .tgz (user-approved 2026-10-03; receipts kept)
  cleanup   delete camera archives whose needed images are already verified (pilot/X1/navtest)
  finalize  merge fragments, check coverage
Scorer environment. Head-val and navtest-dev/final labels are never read here.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import shutil
import socket
import tarfile
import threading
import time
import urllib.request

import numpy as np

from remote_navsim_extract import copy_member, write_json

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
ARCHIVES = ROOT / "data/archives"
SPLITS = ROOT / "splits"
FRAG = ROOT / "reports/full-processed"
STATUS = ROOT / "reports/full-stream-status.json"
LOGS = ROOT / "data/navsim_logs/trainval"
BLOBS = ROOT / "data/sensor_blobs/trainval"
OUT = ROOT / "processed/navtrain-current-v1"
CHUNK = 64 * 1024**2
lock = threading.Lock()
progress = {}

_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET if f == socket.AF_UNSPEC else f, t, pr, fl)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for b in iter(lambda: s.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()

def status(stage, **extra):
    with lock:
        write_json(STATUS, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})

def select():
    assert not (SPLITS / "full-extra-plan.json").exists(), "Selection frozen"
    pilot = {json.loads(l)["scene_token"] for l in (SPLITS / "pilot-manifest.jsonl").read_text().splitlines()
             if json.loads(l)["split"] == "head_train"}
    x1 = set((SPLITS / "x1-head-train-tokens.txt").read_text().splitlines())
    index = [json.loads(l) for l in (SPLITS / "navtrain-scene-index.jsonl").read_text().splitlines()]
    train = [r for r in index if r["split"] == "head_train"]
    extra = [r for r in train if r["scene_token"] not in pilot | x1]
    for r in extra:
        r["split"] = "full_head_train"
        r["input_path"] = str(OUT / "inputs" / (r["scene_token"] + ".npz"))
        r["image_relative"] = str(Path(r["image_path"]).relative_to(BLOBS))
    archives = json.loads((ROOT / "reports/openscene_sensor_trainval_0-199.json").read_text())
    log_to_archive = {log: name for name, logs in archives.items() for log in logs}
    repository = json.loads((ROOT / "reports/openscene-repository.json").read_text())
    files = {s["rfilename"].split("/")[-1]: s for s in repository["siblings"]}
    plan = {}
    for r in extra:
        name = log_to_archive[r["log_name"]].replace("openscene_sensor_trainval_", "openscene_sensor_trainval_camera_") + ".tgz"
        plan.setdefault(name, set()).add(r["log_name"])
    specs = []
    for name in sorted(plan, key=lambda n: (not (ARCHIVES / n).exists(), int(n.split("_")[-1][:-4]))):
        s = files[name]
        specs.append({"name": name, "bytes": s["lfs"]["size"], "publisher_sha256": s["lfs"]["sha256"], "logs": sorted(plan[name]),
                      "url": f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{repository['sha']}/{s['rfilename']}?download=true"})
    (SPLITS / "full-extra-candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in extra))
    write_json(SPLITS / "full-extra-plan.json", {"rule": "pre-registration §7", "head_train_total": len(train), "pilot": len(pilot),
        "x1": len(x1), "extra": len(extra), "archives": specs, "on_disk": sum((ARCHIVES / s["name"]).exists() for s in specs),
        "new_download_bytes": sum(s["bytes"] for s in specs if not (ARCHIVES / s["name"]).exists()),
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"head_train": len(train), "extra": len(extra), "archives": len(specs),
                      "new_GB": sum(s["bytes"] for s in specs if not (ARCHIVES / s["name"]).exists()) / 1e9}))

def download(spec):
    dest = ARCHIVES / spec["name"]
    receipt = dest.with_suffix(dest.suffix + ".json")
    if dest.exists() and receipt.exists():
        return
    pieces = ARCHIVES / (spec["name"] + ".parts"); pieces.mkdir(exist_ok=True)
    chunks = [(i, s, min(s + CHUNK, spec["bytes"]) - 1) for i, s in enumerate(range(0, spec["bytes"], CHUNK))]
    def get(chunk):
        i, first, last = chunk
        path = pieces / f"{i:06d}"; expected = last - first + 1
        if path.exists() and path.stat().st_size == expected:
            return
        for attempt in range(8):
            try:
                req = urllib.request.Request(spec["url"], headers={"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    assert resp.status == 206 and resp.headers.get("Content-Range") == f"bytes {first}-{last}/{spec['bytes']}"
                    part = path.with_suffix(".partial")
                    with part.open("wb") as out:
                        size = 0
                        for block in iter(lambda: resp.read(1024**2), b""):
                            size += len(block); out.write(block)
                    assert size == expected
                    part.replace(path)
                return
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get, chunks))
    h = hashlib.sha256(); partial = dest.with_suffix(dest.suffix + ".partial")
    with partial.open("wb") as out:
        for i, _, _ in chunks:
            with (pieces / f"{i:06d}").open("rb") as s:
                for b in iter(lambda: s.read(16 * 1024**2), b""):
                    h.update(b); out.write(b)
    assert partial.stat().st_size == spec["bytes"] and h.hexdigest() == spec["publisher_sha256"], "Publisher SHA-256 mismatch"
    partial.replace(dest); shutil.rmtree(pieces)
    write_json(receipt, {**{k: spec[k] for k in ["name", "bytes", "url", "publisher_sha256"]}, "path": str(dest), "sha256": h.hexdigest(),
                         "publisher_sha256_verified": True, "completed_at_utc": datetime.now(timezone.utc).isoformat()})

def process_rows(rows):
    import cv2
    from PIL import Image
    from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
    cv2.setNumThreads(1)
    no_sensors = SensorConfig.build_no_sensors()
    done, failed = [], []
    by_log = {}
    for r in rows:
        by_log.setdefault(r["log_name"], []).append(r)
    for log, samples in by_log.items():
        frames = pickle.load((LOGS / (log + ".pkl")).open("rb"))
        for row in samples:
            try:
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
                    assert image.mode == "RGB"; rgb = np.asarray(image)
                h, w = rgb.shape[:2]; ego = agent.ego_statuses[-1]
                values = {"rgb_current": cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA), "history_poses": history,
                          "ego_velocity_acc": np.concatenate([ego.ego_velocity, ego.ego_acceleration]).astype(np.float32),
                          "command_id": np.asarray(int(np.argmax(ego.driving_command)), dtype=np.int64)}
                assert int(values["command_id"]) == row["command_id"]
                assert all(np.isfinite(v).all() for v in values.values()) and np.isfinite(future).all()
                np.savez(row["input_path"], **values)
                np.savez(row["label_path"], future_poses_5s=future, future_poses_4s=future[:8], action_deltas_4s=deltas)
                row.update(input_available=True, target_available=True, input_sha256=sha(row["input_path"]), target_sha256=sha(row["label_path"]))
                done.append(row)
            except Exception as error:
                failed.append({"scene_token": row["scene_token"], "error": repr(error)})
        del frames
    return done, failed

def run(workers):
    plan = json.loads((SPLITS / "full-extra-plan.json").read_text())
    rows = [json.loads(l) for l in (SPLITS / "full-extra-candidates.jsonl").read_text().splitlines()]
    FRAG.mkdir(exist_ok=True)
    (OUT / "inputs").mkdir(parents=True, exist_ok=True); (OUT / "targets").mkdir(parents=True, exist_ok=True)
    finished = []
    def one(spec):
        name = spec["name"]
        frag = FRAG / (name + ".jsonl")
        mine = [r for r in rows if r["log_name"] in set(spec["logs"])]
        if not frag.exists():
            download(spec)
            path = ARCHIVES / name
            needed = {r["image_relative"] for r in mine}
            with tarfile.open(path, "r|gz") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    assert not member.name.startswith("/") and ".." not in parts
                    if not member.isfile() or "CAM_F0" not in parts:
                        continue
                    relative = str(PurePosixPath(*parts[parts.index("CAM_F0") - 1:]))
                    if relative in needed:
                        with archive.extractfile(member) as stream:
                            copy_member(stream, BLOBS / relative, member.size)
            done, failed = process_rows([r for r in mine if (BLOBS / r["image_relative"]).is_file()])
            missing = [r["scene_token"] for r in mine if not (BLOBS / r["image_relative"]).is_file()]
            tmp = frag.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in done)); tmp.replace(frag)
            write_json(FRAG / (name + ".failures.json"), {"missing_images": missing, "processing_failures": failed})
        (ARCHIVES / name).unlink(missing_ok=True)                      # user-approved deletion; receipt kept
        with lock:
            finished.append(name)
        status("streaming", archives_done=len(finished), archives_total=len(plan["archives"]))
    status("streaming", archives_done=0, archives_total=len(plan["archives"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, plan["archives"]))
    status("streamed", archives_done=len(finished), archives_total=len(plan["archives"]))

def cleanup():
    deleted = []
    for path in sorted(ARCHIVES.glob("openscene_sensor_t*_camera_*.tgz")):
        if path.name.startswith("openscene_sensor_test_camera_"):
            report = json.loads((ROOT / "reports/navtest-camera-extraction.json").read_text())
            assert report["missing"] == 0 and report["front_images"] == 12146
        else:
            receipt = ROOT / "reports/x1-camera-extraction" / (path.name + ".json")
            pilot = ROOT / "reports/navsim-camera-extraction" / (path.name + ".json")
            frag = FRAG / (path.name + ".jsonl")
            assert receipt.exists() or pilot.exists() or frag.exists(), f"No verified extraction for {path.name}"
        size = path.stat().st_size
        path.unlink(); deleted.append({"archive": path.name, "bytes": size})
    write_json(ROOT / "reports/full-archive-cleanup.json", {"deleted": deleted, "bytes": sum(d["bytes"] for d in deleted),
               "approved_by_user": "2026-10-03", "receipts_kept": True, "utc": datetime.now(timezone.utc).isoformat()})
    print("deleted", len(deleted), sum(d["bytes"] for d in deleted) / 1e9, "GB")

def finalize():
    plan = json.loads((SPLITS / "full-extra-plan.json").read_text())
    rows, missing, failed = [], [], []
    for spec in plan["archives"]:
        rows += [json.loads(l) for l in (FRAG / (spec["name"] + ".jsonl")).read_text().splitlines()]
        f = json.loads((FRAG / (spec["name"] + ".failures.json")).read_text())
        missing += f["missing_images"]; failed += f["processing_failures"]
    assert len(rows) == len({r["scene_token"] for r in rows})
    excluded = (len(missing) + len(failed)) / plan["extra"]
    (SPLITS / "full-extra-manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (SPLITS / "full-extra-tokens.txt").write_text("".join(r["scene_token"] + "\n" for r in rows))
    write_json(ROOT / "reports/full-extra-processing.json", {"candidates": plan["extra"], "processed": len(rows),
        "missing_images": len(missing), "processing_failures": len(failed), "excluded_fraction": excluded,
        "within_preregistered_0.5pct": excluded <= .005, "utc": datetime.now(timezone.utc).isoformat()})
    status("verified" if excluded <= .005 else "excluded_over_limit", processed=len(rows), excluded_fraction=excluded)
    print(json.dumps({"processed": len(rows), "missing": len(missing), "failed": len(failed), "excluded_fraction": excluded}))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=["select", "run", "cleanup", "finalize"]); p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    try:
        {"select": select, "run": lambda: run(a.workers), "cleanup": cleanup, "finalize": finalize}[a.stage]()
    except Exception as error:
        status("failed", operation=a.stage, error=repr(error)); raise
