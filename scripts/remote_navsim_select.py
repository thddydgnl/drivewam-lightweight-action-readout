"""Choose bounded public camera shards by fixed metadata counts, never scores."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
repository = json.loads((ROOT / "reports/openscene-repository.json").read_text())
mapping = json.loads((ROOT / "reports/openscene_sensor_trainval_0-199.json").read_text())
split = json.loads((ROOT / "splits/navtrain-log-split-seed17.json").read_text())
counts = json.loads((ROOT / "reports/navsim-scene-index.json").read_text())["per_log_scene_counts"]
train, val = set(split["head_train"]), set(split["head_val"])
files = {f["rfilename"]: f for f in repository["siblings"]}
candidates = []
for key, logs in mapping.items():
    index = int(key.rsplit("_", 1)[1])
    filename = f"openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{index}.tgz"
    entry = files[filename]
    selected_logs = set(logs).intersection(train | val)
    if not selected_logs:
        continue
    candidates.append({"index": index, "logs": sorted(selected_logs), "entry": entry})

selected, available_train, available_val = [], set(), set()
target_train, target_val = 4000, 1000
while sum(counts[l] for l in available_train) < target_train or sum(counts[l] for l in available_val) < target_val:
    missing_train = max(0, target_train - sum(counts[l] for l in available_train))
    missing_val = max(0, target_val - sum(counts[l] for l in available_val))
    def utility(candidate):
        added_train = sum(counts[l] for l in set(candidate["logs"]) & train - available_train)
        added_val = sum(counts[l] for l in set(candidate["logs"]) & val - available_val)
        # Tenfold val weight compensates for the global 90/10 log split. Once
        # covered, its contribution is zero. Bytes limit network/disk demand.
        gain = min(added_train, missing_train) + 10 * min(added_val, missing_val)
        return gain / candidate["entry"]["size"], -candidate["index"]
    best = max(candidates, key=utility)
    assert utility(best)[0] > 0, "Cannot satisfy pilot coverage with published shards"
    selected.append(best)
    candidates.remove(best)
    available_train.update(set(best["logs"]) & train)
    available_val.update(set(best["logs"]) & val)

specs = []
for candidate in selected:
    entry = candidate["entry"]
    specs.append({"name": Path(entry["rfilename"]).name, "bytes": entry["size"],
                  "url": f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{repository['sha']}/{entry['rfilename']}?download=true",
                  "publisher_sha256": entry["lfs"]["sha256"], "etag": None})
assert sum(s["bytes"] for s in specs) < 160 * 1024**3, "Pilot download unexpectedly large"
report = {"openscene_revision": repository["sha"], "selected_shards": [c["index"] for c in selected],
          "camera_archives": len(selected), "download_bytes": sum(s["bytes"] for s in specs),
          "head_train_logs": sorted(available_train), "head_val_logs": sorted(available_val),
          "available_head_train_scenes": sum(counts[l] for l in available_train),
          "available_head_val_scenes": sum(counts[l] for l in available_val),
          "pilot_target_head_train": target_train, "pilot_target_head_val": target_val,
          "selection": "greedy metadata scene coverage per archive byte with 10x val weight; tie by lower shard id",
          "selection_uses_model_outputs_or_scores": False,
          "scope_limit": "pilot archive cohort; not the complete navtrain training set",
          "created_at_utc": datetime.now(timezone.utc).isoformat()}
for path, value in [(ROOT / "reports/navsim-camera-download-specs.json", specs),
                    (ROOT / "reports/navsim-pilot-archive-selection.json", report)]:
    path.write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(report, indent=2))
