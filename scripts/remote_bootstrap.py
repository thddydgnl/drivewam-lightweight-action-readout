"""Create an isolated research workspace over the configured SSH relay.

Run with system Python on the GPU host. No shell profile or global packages change.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
UV_VERSION = "0.12.21"
for name in ("tools", "scripts", "upstream", "work", "checkpoints", "data",
             "caches/prompt", "caches/features", "caches/metrics", "caches/uv",
             "caches/huggingface", "locks", "logs", "reports", ".venvs", "python",
             "data/maps", "data/navsim_logs", "data/sensor_blobs", "splits", "processed", "runs"):
    (ROOT / name).mkdir(parents=True, exist_ok=True)

def run(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

uv = ROOT / "tools/uv"
if not uv.exists():
    metadata = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/uv/{UV_VERSION}/json", timeout=30))
    candidates = [f for f in metadata["urls"] if f["filename"].endswith(".whl")
                  and "manylinux" in f["filename"] and "x86_64" in f["filename"]]
    artifact = sorted(candidates, key=lambda f: ("manylinux2014" not in f["filename"], f["filename"]))[0]
    payload = urllib.request.urlopen(artifact["url"], timeout=120).read()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == artifact["digests"]["sha256"], "uv download hash mismatch"
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        member = next(n for n in wheel.namelist() if n.endswith("/uv"))
        uv.write_bytes(wheel.read(member))
    uv.chmod(0o755)
    (ROOT / "locks/uv-bootstrap.json").write_text(json.dumps({
        "version": metadata["info"]["version"], "url": artifact["url"],
        "wheel_sha256": digest, "binary_sha256": hashlib.sha256(uv.read_bytes()).hexdigest()
    }, indent=2) + "\n")

profile = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "connection": {"target_alias": "kiis-mvf-gpu", "relay_alias": "kiis-mvf-campus"},
    "hostname": platform.node(), "user": os.environ.get("USER"),
    "workspace": str(ROOT), "system_python": platform.python_version(),
    "os_release": Path("/etc/os-release").read_text(),
    "kernel": platform.release(), "cpu_logical": os.cpu_count(),
    "cpu_affinity_count": len(os.sched_getaffinity(0)),
    "lscpu": json.loads(run(["lscpu", "--json"])),
    "cgroup": {key: (Path("/sys/fs/cgroup") / key).read_text().strip()
               for key in ("cpu.max", "memory.max") if (Path("/sys/fs/cgroup") / key).exists()},
    "memory": Path("/proc/meminfo").read_text().splitlines()[:3],
    "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
    "gpu_csv": run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader,nounits"]),
    "cuda_compiler": run(["/usr/local/cuda/bin/nvcc", "--version"]),
    "uv": run([str(uv), "--version"]),
    "storage_note": "Home is on an overlay filesystem; retention across container replacement is not verified."
}
(ROOT / "reports/server-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
print(json.dumps(profile, indent=2))
