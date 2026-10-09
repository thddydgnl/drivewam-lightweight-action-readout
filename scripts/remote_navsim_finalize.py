"""Freeze verified dataset preparation evidence; do not infer driving quality."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

if __name__ == "__main__":
    names = ["navsim-download-manifest", "navsim-maps-extraction", "navsim-metadata-extraction",
             "navsim-scene-index", "navsim-pilot-archive-selection", "navsim-pilot-selection",
             "navsim-camera-extraction", "navsim-pilot-processing", "navsim-devval-cache",
             "navsim-reference-parity", "navsim-real-b10-smoke", "navsim-s3-superseded"]
    reports = {name: json.loads((ROOT / "reports" / (name + ".json")).read_text()) for name in names}
    assert reports["navsim-scene-index"]["scenes"] == 103288
    assert reports["navsim-pilot-processing"]["samples"] == 5000
    assert reports["navsim-camera-extraction"]["missing_required_images"] == 0
    assert reports["navsim-devval-cache"]["coverage"] == 1.0
    assert reports["navsim-real-b10-smoke"]["outputs_finite"]
    assert reports["navsim-real-b10-smoke"]["reference_with_real_future_rgb_and_gt_actions_has_bitwise_identical_predictions"]
    assert reports["navsim-real-b10-smoke"]["driving_quality_evaluated"] is False
    cache_tokens = set(reports["navsim-devval-cache"]["requested_tokens"])
    manifest = ROOT / "splits/pilot-manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        assert row["input_available"] and row["target_available"]
        row["metric_cache_available"] = row["scene_token"] in cache_tokens
    assert sum(row["metric_cache_available"] for row in rows) == 256
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scripts = ["remote_navsim_download.py", "remote_navsim_extract.py", "remote_navsim_select.py",
               "remote_navsim_pilot.py", "remote_navsim_metric_cache.py", "navsim_current_dataset.py",
               "remote_navsim_reference.py", "remote_navsim_real_smoke.py", "remote_navsim_finalize.py"]
    frozen_files = [ROOT / "scripts" / name for name in scripts]
    frozen_files += [path for path in sorted((ROOT / "splits").iterdir()) if path.is_file()]
    frozen_files += [ROOT / "reports" / (name + ".json") for name in names]
    frozen_files += [ROOT / "reports/navsim-data-LICENSE.txt", ROOT / "data/maps/LICENSE"]
    frozen_files += [ROOT / "reports" / name for name in ["openscene-repository.json",
        "openscene_sensor_trainval_0-199.json", "openscene_sensor_mini_0-31.json",
        "navsim-camera-download-specs.json", "navsim-pilot-required-images.json"]]
    # Only this task's abandoned download pieces are cleaned up, after the
    # replacement assets and real-scene check have all passed.
    abandoned = ROOT / "data/archives/navtrain_current_1.tgz.parts"
    removed = sum(path.stat().st_size for path in abandoned.iterdir() if path.is_file()) if abandoned.exists() else 0
    if abandoned.exists():
        shutil.rmtree(abandoned)
    archives = reports["navsim-download-manifest"]["files"]
    record = {"as_of": datetime.now(timezone.utc).isoformat(),
       "status": "navtrain_pilot_prepared_and_real_B10_smoke_verified",
       "workspace": str(ROOT), "environment_lock_sha256": sha(ROOT / "ENVIRONMENT_LOCK.json"),
       "openscene_revision": reports["navsim-download-manifest"]["openscene_revision"],
       "map_version": "nuplan-maps-v1.0", "archive_files": archives,
       "archive_download_bytes": sum(file["bytes"] for file in archives),
       "reports": {name: {"path": str(ROOT / "reports" / (name + ".json")),
                          "sha256": sha(ROOT / "reports" / (name + ".json"))} for name in names},
       "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in frozen_files],
       "prepared": {"navtrain_metadata_logs": 1192, "navtrain_metadata_scenes": 103288,
           "head_train": 4000, "head_val": 1000, "smoke_train": 32, "devval": 256,
           "devval_metric_cache": 256, "reference_fixtures": 1, "real_B10_smoke_scenes": 1,
           "navtest_data_or_scores_used": False},
       "split": {"seed": 17, "unit": "log_name", "full_head_train_logs": 1072,
                 "full_head_val_logs": 120, "train_val_log_overlap": 0, "train_val_token_overlap": 0},
       "manifest_sha256": sha(manifest), "discarded_superseded_partial_bytes": removed,
       "disk_free_bytes": shutil.disk_usage(ROOT).free,
       "scope": "Full navtrain metadata and a fixed 4k/1k pilot. Complete full-train RGB acquisition is pending.",
       "remaining": ["B10/B05/opt quality comparison on dev/head-val",
                     "P01-P10 remaining research evaluator/feature/scoring integration",
                     "Head implementation, feature extraction and training",
                     "Remaining full navtrain camera data and locked navtest data acquisition",
                     "Storage retention across container replacement"]}
    (ROOT / "DATA_LOCK.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ["status", "prepared", "archive_download_bytes", "disk_free_bytes"]}, indent=2))
