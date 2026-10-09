"""Download a bounded NAVSIM v1.1 pilot with resumable, verified HTTP ranges.

Run on the GPU host with system Python. Public URLs only; no credentials or
system network settings are changed. Raw archives stay under data/archives.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import urllib.request

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
ARCHIVES = ROOT / "data/archives"
ARCHIVES.mkdir(parents=True, exist_ok=True)
original_getaddrinfo = socket.getaddrinfo
def ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family,
                               type, proto, flags)
socket.getaddrinfo = ipv4
repository = json.loads((ROOT / "reports/openscene-repository.json").read_text())
revision = repository["sha"]
assert revision == "a76f840b65e972bc45e56c2adced897498e9a026", "Unexpected OpenScene revision"
metadata = next(f for f in repository["siblings"]
                if f["rfilename"] == "openscene-v1.1/openscene_metadata_trainval.tgz")
FILES = [
    {"name": "nuplan-maps-v1.1.zip", "bytes": 970997691,
     "url": "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip",
     "etag": '"0efaac6b7c603ae6f341021b35059bcc-116"', "publisher_sha256": None},
    {"name": "openscene_metadata_trainval.tgz", "bytes": metadata["size"],
     "url": f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{revision}/{metadata['rfilename']}?download=true",
     "publisher_sha256": metadata["lfs"]["sha256"], "etag": None},
]
parser = argparse.ArgumentParser()
parser.add_argument("--files", type=Path, help="JSON list of exact public URL/size/hash specifications")
args = parser.parse_args()
if args.files:
    FILES = json.loads(args.files.read_text())
CHUNK = 64 * 1024**2
lock = threading.Lock()
progress = {}
started = time.monotonic()
report = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
          "openscene_revision": revision, "files": [],
          "license_copy": str(ROOT / "reports/navsim-data-LICENSE.txt"),
          "scope": "explicit bounded list of NAVSIM archives; front camera extraction only"}
manifest_path = ROOT / "reports/navsim-download-manifest.json"
if manifest_path.exists():
    report["files"] = json.loads(manifest_path.read_text()).get("files", [])
status_path = ROOT / "reports/navsim-download-status.json"

def save_status():
    with lock:
        data = {"stage": "downloading", "elapsed_seconds": round(time.monotonic() - started),
                "files": {key: dict(value) for key, value in progress.items()}}
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(status_path)

def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def download(spec):
    dest = ARCHIVES / spec["name"]
    receipt = dest.with_suffix(dest.suffix + ".json")
    if dest.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        assert dest.stat().st_size == spec["bytes"] and record["url"] == spec["url"]
        assert digest_file(dest) == record["sha256"]
        with lock:
            progress[spec["name"]] = {"stage": "verified", "bytes": spec["bytes"], "total": spec["bytes"]}
        return record
    pieces = ARCHIVES / (spec["name"] + ".parts")
    pieces.mkdir(exist_ok=True)
    chunks = [(i, start, min(start + CHUNK, spec["bytes"]) - 1)
              for i, start in enumerate(range(0, spec["bytes"], CHUNK))]
    with lock:
        progress[spec["name"]] = {"stage": "downloading", "bytes": 0, "total": spec["bytes"]}

    def get_chunk(chunk):
        index, first, last = chunk
        path = pieces / f"{index:06d}"
        expected = last - first + 1
        if not (path.exists() and path.stat().st_size == expected):
            for attempt in range(8):
                try:
                    headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"}
                    if spec["etag"]:
                        headers["If-Match"] = spec["etag"]
                    req = urllib.request.Request(spec["url"], headers=headers)
                    with urllib.request.urlopen(req, timeout=120) as response:
                        assert response.status == 206, f"Expected range response, got {response.status}"
                        expected_range = f"bytes {first}-{last}/{spec['bytes']}"
                        assert response.headers.get("Content-Range") == expected_range, response.headers.get("Content-Range")
                        if spec["etag"]:
                            assert response.headers.get("ETag") == spec["etag"]
                        partial = path.with_suffix(".partial")
                        size = 0
                        with partial.open("wb") as out:
                            for block in iter(lambda: response.read(1024**2), b""):
                                size += len(block)
                                assert size <= expected
                                out.write(block)
                        assert size == expected, (size, expected)
                        partial.replace(path)
                    break
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(min(2 ** attempt, 30))
        with lock:
            progress[spec["name"]]["bytes"] += expected
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=32 if spec["name"] == "navtrain_current_1.tgz" else 8) as pool:
        list(pool.map(get_chunk, chunks))
    with lock:
        progress[spec["name"]]["stage"] = "assembling_and_hashing"
    h = hashlib.sha256()
    partial = dest.with_suffix(dest.suffix + ".partial")
    with partial.open("wb") as output:
        for index, _, _ in chunks:
            with (pieces / f"{index:06d}").open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024**2), b""):
                    h.update(block)
                    output.write(block)
    assert partial.stat().st_size == spec["bytes"]
    sha = h.hexdigest()
    if spec["publisher_sha256"]:
        assert sha == spec["publisher_sha256"], "Publisher SHA-256 mismatch"
    partial.replace(dest)
    record = {**spec, "path": str(dest), "sha256": sha,
              "publisher_sha256_verified": bool(spec["publisher_sha256"]),
              "etag_verified_per_range": bool(spec["etag"]),
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    # Only this downloader's temporary, verified chunk files are removed.
    shutil.rmtree(pieces)
    with lock:
        progress[spec["name"]]["stage"] = "verified"
    print(json.dumps(record), flush=True)
    return record

assert shutil.disk_usage(ROOT).free > 220 * 1024**3, "Need 220 GiB free for download and extraction staging"
stop = threading.Event()
def heartbeat():
    while not stop.wait(10):
        save_status()
thread = threading.Thread(target=heartbeat, daemon=True)
thread.start()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, spec) for spec in FILES]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            report["files"] = [f for f in report["files"] if f["name"] != record["name"]] + [record]
            (ROOT / "reports/navsim-download-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (ROOT / "reports/navsim-download-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    save_status()
    status = json.loads(status_path.read_text())
    status["stage"] = "verified"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
except Exception as exc:
    save_status()
    status = json.loads(status_path.read_text())
    status.update(stage="failed", error=repr(exc))
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    raise
finally:
    stop.set()
    thread.join()
