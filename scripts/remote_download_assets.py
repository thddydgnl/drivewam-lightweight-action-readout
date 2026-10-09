"""Download and hash only the pinned NAVSIM model, base VAE/text, and scorer cache."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
from datetime import datetime, timezone
from huggingface_hub import snapshot_download

# This host has unrouted IPv6. Restrict only this download process to IPv4.
original_getaddrinfo = socket.getaddrinfo
def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family,
                                type, proto, flags)
socket.getaddrinfo = ipv4_getaddrinfo

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
manifest = json.loads((ROOT / "scripts/SOURCE_LOCK.json").read_text())
status_file = ROOT / "reports/assets-status.json"
report = {"files": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}
free = shutil.disk_usage(ROOT).free
assert free > 80 * 1024**3, "Need at least 80 GiB free before downloading assets"

for asset in manifest["model_assets"]:
    drive = asset["repo_id"] == "chenchenshi/DriveWAM"
    paths = (["navsim/*", "metric_cache_navtest.tar.gz", "train_4k.csv"] if drive
             else ["vae/*", "text_encoder/*", "tokenizer/*"])
    destination = ROOT / "checkpoints" / ("drivewam" if drive else "lingbot-va-base")
    status_file.write_text(json.dumps({"stage": "downloading", "repo": asset["repo_id"],
                                      "revision": asset["revision"]}, indent=2) + "\n")
    snapshot_download(repo_id=asset["repo_id"], revision=asset["revision"],
                      allow_patterns=paths, local_dir=destination, max_workers=4)
    selected = [f for f in asset["files"] if (drive and (f["path"].startswith("navsim/")
                  or f["path"] in ("metric_cache_navtest.tar.gz", "train_4k.csv")))
                or (not drive and f["path"].split("/")[0] in ("vae", "text_encoder", "tokenizer"))]
    for entry in selected:
        path = destination / entry["path"]
        assert path.stat().st_size == entry["bytes"], f"Size mismatch: {path}"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
                digest.update(chunk)
        computed = digest.hexdigest()
        if entry["publisher_sha256"]:
            assert computed == entry["publisher_sha256"], f"SHA256 mismatch: {path}"
        record = {"repo_id": asset["repo_id"], "revision": asset["revision"],
                  "path": str(path), "bytes": entry["bytes"], "sha256": computed,
                  "publisher_sha256_verified": bool(entry["publisher_sha256"])}
        report["files"].append(record)
        (ROOT / "reports/assets-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)

layout = ROOT / "checkpoints/drivewam-navsim"
layout.mkdir(exist_ok=True)
link = layout / "transformer"
if not link.exists():
    link.symlink_to(ROOT / "checkpoints/drivewam/navsim", target_is_directory=True)
report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
(ROOT / "reports/assets-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
status_file.write_text(json.dumps({"stage": "verified", "files": len(report["files"]),
                                  "bytes": sum(f["bytes"] for f in report["files"])}, indent=2) + "\n")
