"""Pre-registered log-level navtest split into navtest-dev and locked navtest-final.

Uses only the pinned navtest.yaml log/token lists. No sensor data, prediction,
label or score is read. Run once, before any navtest inference, in the scorer
environment (needs PyYAML). Existing outputs are never overwritten.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import yaml

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
FILTER = ROOT / "upstream/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
SPLITS = ROOT / "splits"
SEED = 17
DEV_FRACTION = 0.25

if __name__ == "__main__":
    out = SPLITS / "navtest-log-split-seed17.json"
    assert not out.exists(), "Split already frozen; never redraw"
    navtest = yaml.safe_load(FILTER.read_text())
    train = yaml.safe_load((FILTER.parent / "navtrain.yaml").read_text())
    logs = sorted(navtest["log_names"])
    assert len(logs) == len(set(logs)) == 136 and len(navtest["tokens"]) == 12146
    assert not set(logs) & set(train["log_names"]) and not set(navtest["tokens"]) & set(train["tokens"])
    shuffled = list(logs)
    random.Random(SEED).shuffle(shuffled)
    n_dev = math.ceil(len(logs) * DEV_FRACTION)
    dev, final = sorted(shuffled[:n_dev]), sorted(shuffled[n_dev:])
    record = {
        "seed": SEED, "dev_fraction": DEV_FRACTION,
        "algorithm": "random.Random(17).shuffle(sorted(navtest log_names)); first ceil(25%) logs -> navtest_dev",
        "navtest_yaml_sha256": hashlib.sha256(FILTER.read_bytes()).hexdigest(),
        "navtest_dev_logs": dev, "navtest_final_logs": final,
        "navtest_final_locked": True,
        "token_assignment": "scene tokens are assigned by their log after metadata indexing",
        "created_before_any_navtest_prediction": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SPLITS.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"dev_logs": len(dev), "final_logs": len(final),
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
