"""Collect verified setup evidence after all runtime checks have passed."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
names = ["server-profile", "source-verification", "assets-manifest", "assets-status",
         "model-smoke", "nccl-smoke", "scorer-smoke", "blas-diagnostic",
         "model-blas-smoke", "metric-cache-smoke", "checkpoint-smoke", "prompt-cache-readback"]
reports = {name: json.loads((ROOT / "reports" / (name + ".json")).read_text()) for name in names}
assert reports["assets-status"]["stage"] == "verified"
assert reports["nccl-smoke"]["passed"]
assert reports["scorer-smoke"]["pinv_identity_max_abs_error"] < 1e-12
assert reports["checkpoint-smoke"]["synthetic_b10"]["outputs_finite"]
assert reports["checkpoint-smoke"]["synthetic_b10"]["driving_quality_evaluated"] is False
for role in ("model", "scorer"):
    state = json.loads((ROOT / "reports" / (role + "-install-status.json")).read_text())
    assert state["stage"] == "installed", state
files = []
for filename in ("model-freeze.txt", "scorer-freeze.txt", "scorer-input.txt", "scorer-constraints.txt",
                 "native-libs.json", "uv-bootstrap.json", "network-dns-cache.json"):
    path = ROOT / "locks" / filename
    files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
record = {"as_of": datetime.now(timezone.utc).isoformat(),
          "status": "remote_environment_and_synthetic_B10_verified",
          "ssh": {"relay_alias": "kiis-mvf-campus", "gpu_alias": "kiis-mvf-gpu"},
          "workspace": str(ROOT), "reports": reports, "dependency_locks": files,
          "setup_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in sorted((ROOT / "scripts").iterdir()) if path.is_file()],
          "flash_attention_build": {"version": "2.8.3", "source": "PyPI source distribution",
              "torch": "2.9.0+cu126", "nvcc": "12.4.131", "cuda_archs": "80",
              "max_jobs": 24, "nvcc_threads": 2, "cuda_minor_versions_differ": True,
              "validation": "Both GPU forward/reference and finite backward checks passed"},
          "remaining": ["NAVSIM maps/logs/RGB acquisition and preprocessing",
                        "P01-P10 research evaluator/observation/scoring patches",
                        "B10 on an actual NAVSIM observation",
                        "Driving-quality evaluation and head training",
                        "Storage retention across container replacement"]}
(ROOT / "ENVIRONMENT_LOCK.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": record["status"], "lock": str(ROOT / "ENVIRONMENT_LOCK.json"),
                  "remaining": record["remaining"]}, indent=2))
