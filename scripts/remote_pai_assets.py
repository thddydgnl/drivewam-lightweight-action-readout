"""PhysicalAI extension X0: pinned DriveWAM PhysicalAI assets + NVIDIA clip index.

Downloads (approved 2026-10-04): DriveWAM@c4fb6d2c physicalai checkpoint/config,
train_{4k,20k,100k}.csv, nav prompts tarball; NVIDIA PhysicalAI-AV clip_index.parquet.
SHA-256 is verified against the pinned LFS hashes where published. The HF token is
read by huggingface_hub from HF_HOME and never printed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
DRIVEWAM_REV = "c4fb6d2c5d26ee3018a9ee1507f36fd02b85b1ce"
PINNED = {
    "physicalai/diffusion_pytorch_model.safetensors": "ad90feae1a26007d13743ecb7896887ae7f9906168e5652d57b5406073fa0d98",
    "train_100k.csv": "06338eefac9a9214e7010c2597a7642768f7e11b3718ebd2d9247e1207d914dd",
    "PhysicalAI-Autonomous-Vehicles-Front-nav_prompts.tar.gz": "8753c19acc318d963d6378e324d0a9a4dc06ee9f60a8b79d4a2608840c359d3b",
}
FILES = ["physicalai/config.json", "physicalai/diffusion_pytorch_model.safetensors", "train_4k.csv", "train_20k.csv",
         "train_100k.csv", "PhysicalAI-Autonomous-Vehicles-Front-nav_prompts.tar.gz"]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for b in iter(lambda: s.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()

if __name__ == "__main__":
    report = {"drivewam_revision": DRIVEWAM_REV, "files": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}
    for name in FILES:
        path = hf_hub_download("chenchenshi/DriveWAM", name, revision=DRIVEWAM_REV, local_dir=str(ROOT / "checkpoints/drivewam"))
        digest = sha(path)
        if name in PINNED:
            assert digest == PINNED[name], f"SHA-256 mismatch: {name}"
        report["files"].append({"name": name, "path": path, "bytes": Path(path).stat().st_size, "sha256": digest,
                                "pinned_verified": name in PINNED})
        print("ok", name, Path(path).stat().st_size, flush=True)
    link = ROOT / "checkpoints/drivewam-physicalai/transformer"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(ROOT / "checkpoints/drivewam/physicalai", target_is_directory=True)
    api = HfApi()
    info = api.dataset_info("nvidia/PhysicalAI-Autonomous-Vehicles", files_metadata=False)
    nv_rev = info.sha
    path = hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles", "clip_index.parquet", repo_type="dataset",
                           revision=nv_rev, local_dir=str(ROOT / "data/physicalai"))
    report["nvidia_dataset_revision"] = nv_rev
    report["files"].append({"name": "clip_index.parquet", "path": path, "bytes": Path(path).stat().st_size, "sha256": sha(path)})
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    out = ROOT / "reports/pai-assets.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("done; nvidia revision", nv_rev, flush=True)
