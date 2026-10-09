"""Extract the official Ubuntu GL runtime into this project only."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
debs = ROOT / "tools/debs"
debs.mkdir(exist_ok=True)
runtime = ROOT / "tools/runtime-libs"
runtime.mkdir(exist_ok=True)
subprocess.run(["apt-get", "download", "libgl1=1.4.0-1", "libglx0=1.4.0-1", "libglvnd0=1.4.0-1"],
               cwd=debs, check=True)
records = []
for file in sorted(debs.glob("*.deb")):
    subprocess.run(["dpkg-deb", "-x", str(file), str(runtime)], check=True)
    records.append({"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
(ROOT / "locks/native-libs.json").write_text(json.dumps({"packages": records,
    "extracted_to": str(runtime), "installation": "project-local extraction; system packages unchanged"}, indent=2) + "\n")
