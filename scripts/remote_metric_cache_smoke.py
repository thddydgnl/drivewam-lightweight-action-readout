"""Check a hash-verified metric archive's serialization; do not compute test scores."""
import hashlib
import json
import lzma
from pathlib import Path
import pickle
import tarfile
from navsim.planning.metric_caching.metric_cache import MetricCache

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
records = json.loads((ROOT / "reports/assets-manifest.json").read_text())["files"]
archive = ROOT / "checkpoints/drivewam/metric_cache_navtest.tar.gz"
assert any(r["path"] == str(archive) and r["publisher_sha256_verified"] for r in records)
destination = ROOT / "caches/metrics/smoke/metric_cache.pkl"
destination.parent.mkdir(exist_ok=True)
with tarfile.open(archive, "r|gz") as stream:
    for member in stream:
        if member.isfile() and member.name.endswith("/metric_cache.pkl"):
            destination.write_bytes(stream.extractfile(member).read())
            record = {"archive_member": member.name, "local_path": str(destination),
                      "bytes": member.size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
            break
    else:
        raise RuntimeError("No metric cache found")
with lzma.open(destination, "rb") as stream:
    cache = pickle.load(stream)
assert isinstance(cache, MetricCache)
for key in ("trajectory", "ego_state", "observation", "centerline", "route_lane_ids", "drivable_area_map"):
    assert getattr(cache, key) is not None
record.update({"deserialization": "passed", "navtest_scores_computed": False})
(ROOT / "reports/metric-cache-smoke.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
