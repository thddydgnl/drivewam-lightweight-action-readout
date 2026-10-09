"""Round-3 X1: 16,000 additional navtrain head-train scenes (pre-registered §6).

  select   archives in random.Random(17) order until >=16,000 available head-train
           scenes outside pilot logs; pick 16,000 by SHA256("17:x1:"+token)
  extract  current-frame CAM_F0 images only, from verified archives
  process  observation-only inputs and separate targets (same code path as pilot)
Run in the scorer environment. Head-val and navtest are never touched.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import tarfile
import time

from remote_navsim_extract import copy_member, write_json, verified_archive

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
SPLITS = ROOT / "splits"
STATUS = ROOT / "reports/x1-prepare-status.json"
TARGET = 16000

def select():
    out = SPLITS / "x1-manifest.jsonl"
    assert not out.exists(), "X1 selection already frozen"
    pilot_logs = {json.loads(l)["log_name"] for l in (SPLITS / "pilot-manifest.jsonl").read_text().splitlines()}
    index = [json.loads(l) for l in (SPLITS / "navtrain-scene-index.jsonl").read_text().splitlines()]
    candidates = {}
    for row in index:
        if row["split"] == "head_train" and row["log_name"] not in pilot_logs:
            candidates.setdefault(row["log_name"], []).append(row)
    archives = json.loads((ROOT / "reports/openscene_sensor_trainval_0-199.json").read_text())
    order = sorted(archives, key=lambda n: int(n.rsplit("_", 1)[1]))
    random.Random(17).shuffle(order)
    chosen, rows = [], []
    for name in order:
        logs = [log for log in archives[name] if log in candidates]
        if not logs:
            continue
        chosen.append(name)
        for log in logs:
            rows.extend(candidates.pop(log))
        if len(rows) >= TARGET:
            break
    assert len(rows) >= TARGET
    rows = sorted(rows, key=lambda r: hashlib.sha256(f"17:x1:{r['scene_token']}".encode()).hexdigest())[:TARGET]
    for row in rows:
        row["split"] = "x1_head_train"
        row["input_path"] = str(ROOT / "processed/navtrain-current-v1/inputs" / (row["scene_token"] + ".npz"))
        row["image_relative"] = str(Path(row["image_path"]).relative_to(ROOT / "data/sensor_blobs/trainval"))
    assert not {r["log_name"] for r in rows} & pilot_logs
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (SPLITS / "x1-head-train-tokens.txt").write_text("".join(r["scene_token"] + "\n" for r in rows))
    repository = json.loads((ROOT / "reports/openscene-repository.json").read_text())
    revision = repository["sha"]
    files = {s["rfilename"].split("/")[-1]: s for s in repository["siblings"]}
    specs = []
    for name in chosen:
        archive = name.replace("openscene_sensor_trainval_", "openscene_sensor_trainval_camera_") + ".tgz"
        sibling = files[archive]
        specs.append({"name": archive, "bytes": sibling["lfs"]["size"], "publisher_sha256": sibling["lfs"]["sha256"], "etag": None,
                      "url": f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{revision}/{sibling['rfilename']}?download=true"})
    write_json(ROOT / "reports/x1-download-specs.json", specs)
    write_json(ROOT / "reports/x1-selection.json", {"archives": chosen, "archive_order_seed": 17, "scenes": len(rows),
        "logs": len({r["log_name"] for r in rows}), "download_bytes_new": sum(s["bytes"] for s in specs if not (ROOT / "data/archives" / s["name"]).exists()),
        "rule": "pre-registration §6", "manifest_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"archives": len(chosen), "scenes": len(rows), "logs": len({r["log_name"] for r in rows}),
                      "new_GB": sum(s["bytes"] for s in specs if not (ROOT / "data/archives" / s["name"]).exists()) / 1e9}))

def extract():
    rows = [json.loads(l) for l in (SPLITS / "x1-manifest.jsonl").read_text().splitlines()]
    needed = {r["image_relative"] for r in rows}
    destination = ROOT / "data/sensor_blobs/trainval"
    specs = json.loads((ROOT / "reports/x1-download-specs.json").read_text())
    receipts = ROOT / "reports/x1-camera-extraction"
    receipts.mkdir(exist_ok=True)

    def one(spec):
        name = spec["name"]
        path = ROOT / "data/archives" / name
        deadline = time.monotonic() + 6 * 3600
        while not path.with_suffix(path.suffix + ".json").exists():
            assert time.monotonic() < deadline, f"Timed out waiting for {name}"
            time.sleep(10)
        path, receipt = verified_archive(name)
        report_path = receipts / (name + ".json")
        if report_path.exists():
            return json.loads(report_path.read_text())
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
                        records.append(copy_member(stream, destination / relative, member.size))
        result = {"archive": receipt, "files": records, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(report_path, result)
        print("extracted", name, len(records), flush=True)
        return result

    write_json(STATUS, {"stage": "extracting", "archives": len(specs), "images": len(needed)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, specs))
    missing = [p for p in needed if not (destination / p).is_file()]
    assert not missing, f"Missing images: {len(missing)}"
    write_json(STATUS, {"stage": "images_verified", "images": len(needed)})

def process():
    import remote_navtest_prepare as base
    base.LOGS = ROOT / "data/navsim_logs/trainval"
    base.BLOBS = ROOT / "data/sensor_blobs/trainval"
    base.OUT = ROOT / "processed/navtrain-current-v1"
    base.STATUS = STATUS
    # Reuse the verified navtest processing path on the X1 manifest rows.
    rows_path = SPLITS / "x1-manifest.jsonl"
    original = base.SPLITS
    temp = ROOT / "splits/.x1-proc"
    temp.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    (temp / "navtest-scene-index.jsonl").write_text("".join(json.dumps({**r, "split": "navtest_dev"}) + "\n" for r in rows))
    base.SPLITS = temp
    # base.process writes ROOT/reports/<split>-processing.json; redirect it so
    # the real navtest-dev processing report is never overwritten.
    (temp / "reports").mkdir(exist_ok=True)
    base.ROOT = temp
    base.process("navtest_dev")
    base.ROOT = ROOT
    (ROOT / "reports/x1-processing.json").write_text((temp / "reports/navtest-dev-processing.json").read_text())
    processed = [json.loads(l) for l in (temp / "navtest-dev-manifest.jsonl").read_text().splitlines()]
    for r in processed:
        r["split"] = "x1_head_train"
    (SPLITS / "x1-manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in processed))
    base.SPLITS = original
    write_json(STATUS, {"stage": "processed_and_verified", "samples": len(processed),
                        "manifest_sha256": hashlib.sha256((SPLITS / "x1-manifest.jsonl").read_bytes()).hexdigest()})
    print("processed", len(processed), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "extract", "process"])
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception as error:
        write_json(STATUS, {"stage": "failed", "operation": args.stage, "error": repr(error)})
        raise
