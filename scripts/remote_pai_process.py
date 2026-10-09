"""PhysicalAI X1: stream clips through the official devkit and reproduce DriveWAM preprocessing.

DriveWAM's `process_clip` (pinned commit abbefb40) is loaded from the pinned source by AST.
The ONLY transformation is `maybe_stream=False` -> `maybe_stream=True`, so the devkit reads the
clip's members from the remote chunk zip instead of downloading the whole ~1.3 GB chunk. The
10 Hz mp4v + ego pkl are produced exactly as upstream, then decoded with upstream
`_load_all_video_frames` (cv2 INTER_AREA to 256x448). We keep every 10th frame (1 fps from
frame 0) plus the frame count, which is all the evaluation loader reads. The 10 Hz mp4 is kept
only when --keep-mp4 (VLM tagging/guidance need it). Devkit env (Python 3.11, cv2 4.10.0.82).
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import threading
import time
import traceback

import cv2
import functools
import numpy as np
import physical_ai_av

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
SRC = ROOT / "work/DriveWAM/src"
NV_REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
OUT = ROOT / "processed/physicalai-v1"
FRONT = ROOT / "data/physicalai/front"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for b in iter(lambda: s.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()

class StreamOn(ast.NodeTransformer):
    changed = 0
    def visit_keyword(self, node):
        if node.arg == "maybe_stream" and isinstance(node.value, ast.Constant) and node.value.value is False:
            StreamOn.changed += 1
            return ast.keyword(arg="maybe_stream", value=ast.Constant(value=True))
        return node

def load_upstream():
    StreamOn.changed = 0
    tree = ast.parse((SRC / "physicalai/process_data.py").read_text())
    keep = {"nearest_indices", "select_camera_10hz_indices", "extract_camera_intrinsics", "process_clip"}
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in
            {"CAM_NAME", "CAMERA_PERIOD_US", "OUTPUT_FPS"} for t in n.targets)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    original = ast.unparse(ast.Module(body=body, type_ignores=[]))
    module = StreamOn().visit(ast.Module(body=body, type_ignores=[]))
    assert StreamOn.changed == 4, StreamOn.changed           # camera, egomotion, extrinsics, intrinsics calls
    ast.fix_missing_locations(module)
    scope = {"np": np, "cv2": cv2, "os": os, "pickle": pickle, "lru_cache": functools.lru_cache}
    exec(compile(module, "<drivewam-process_data-stream>", "exec"), scope)
    data_tree = ast.parse((SRC / "dataset/physicalai_dataset.py").read_text())
    loader = [n for n in data_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_all_video_frames"]
    exec(compile(ast.Module(body=loader, type_ignores=[]), "<drivewam-loader>", "exec"), scope)
    return scope, {"original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                   "transformed_sha256": hashlib.sha256(ast.unparse(module).encode()).hexdigest(),
                   "keyword_changes": StreamOn.changed, "transformation": "maybe_stream=False -> True only"}

WORKER = {}

def init_worker():
    WORKER["scope"], WORKER["provenance"] = load_upstream()
    WORKER["avdi"] = physical_ai_av.PhysicalAIAVDatasetInterface(revision=NV_REVISION,
        cache_dir=str(ROOT / "caches/pai-devkit"), confirm_download_threshold_gb=1.0)

def one(job):
    clip, keep_mp4 = job
    out = OUT / clip
    receipt = out / "receipt.json"
    if receipt.exists():
        if keep_mp4 and not (FRONT / clip / "camera_front_wide_120fov.mp4").exists():
            receipt.unlink()                      # reprocess to recover the 10 Hz mp4
        else:
            return clip, "cached"
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "tmp"))
    try:
        t0 = time.monotonic()
        if not WORKER["scope"]["process_clip"](WORKER["avdi"], clip, str(tmp)):
            return clip, "skip"
        mp4 = tmp / clip / "camera_front_wide_120fov.mp4"
        pkl = tmp / clip / "camera_front_wide_120fov_ego.pkl"
        # Undecorated upstream loader: identical output, no per-path lru_cache growth.
        frames = WORKER["scope"]["_load_all_video_frames"].__wrapped__(str(mp4), 256, 448)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "frames_1fps.npz", frames=np.stack(frames[::10]), n_frames=np.int64(len(frames)), stride=np.int64(10))
        shutil.copy2(pkl, out / "camera_front_wide_120fov_ego.pkl")
        record = {"clip_id": clip, "n_frames_10hz": len(frames), "frames_sha256": sha(out / "frames_1fps.npz"),
                  "ego_sha256": sha(out / "camera_front_wide_120fov_ego.pkl"), "mp4_bytes": mp4.stat().st_size,
                  "seconds": time.monotonic() - t0, "nvidia_revision": NV_REVISION, "provenance": WORKER["provenance"]}
        if keep_mp4:
            dest = FRONT / clip; dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mp4), dest / mp4.name); shutil.copy2(pkl, dest / pkl.name)
            record["mp4_sha256"] = sha(dest / mp4.name)
        tmp_receipt = receipt.with_suffix(".tmp"); tmp_receipt.write_text(json.dumps(record) + "\n"); tmp_receipt.replace(receipt)
        return clip, "done"
    except Exception:
        (ROOT / "logs/pai-process-errors").mkdir(parents=True, exist_ok=True)
        (ROOT / "logs/pai-process-errors" / f"{clip}.txt").write_text(traceback.format_exc())
        return clip, "fail"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="JSON list of [clip_id, keep_mp4] in processing order")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    _, provenance = load_upstream()
    jobs = [tuple(x) for x in json.loads(Path(a.plan).read_text())]
    OUT.mkdir(parents=True, exist_ok=True); (ROOT / "tmp").mkdir(exist_ok=True)
    status_path = ROOT / f"reports/pai-process-{a.tag}-status.json"
    counts, start, failed = {}, time.monotonic(), []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork"), initializer=init_worker) as pool:
        for i, (clip, r) in enumerate(pool.map(one, jobs, chunksize=1), 1):
            counts[r] = counts.get(r, 0) + 1
            if r == "fail":
                failed.append(clip)
            if i % 50 == 0 or i == len(jobs):
                status_path.write_text(json.dumps({"stage": "processing", "tag": a.tag, "done": i, "total": len(jobs),
                    "counts": counts, "elapsed_s": time.monotonic() - start, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    status_path.write_text(json.dumps({"stage": "verified" if not failed else "completed_with_failures", "tag": a.tag,
        "total": len(jobs), "counts": counts, "failed": failed, "elapsed_s": time.monotonic() - start,
        "provenance": provenance, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(json.dumps(counts))
