"""Build a separate official metric cache for all 1,000 pilot head-val scenes."""
from datetime import datetime, timezone
import json
import csv
import hashlib
import shutil
import logging
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import yaml

from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.metric_caching.caching import cache_data
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
CACHE = ROOT / "caches/metrics/pilot-head-val1000"
STATUS = ROOT / "reports/navsim-pilot-val1000-cache-status.json"

def status(stage, **extra):
    STATUS.write_text(json.dumps({"stage": stage, **extra}, indent=2) + "\n")
    print(stage, flush=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tokens = set((ROOT / "splits/pilot-head-val-tokens.txt").read_text().splitlines())
    assert len(tokens) == 1000
    try:
        assert np.max(np.abs(np.linalg.pinv(np.eye(40))-np.eye(40))) < 1e-12
        scene_filter = yaml.safe_load((ROOT / "splits/pilot-head-val.yaml").read_text())
        config = OmegaConf.create({"train_test_split": {"scene_filter": scene_filter},
            "navsim_log_path": str(ROOT / "data/navsim_logs/trainval"),
            "cache": {"cache_path": str(CACHE), "force_feature_computation": False}})
        (ROOT / "reports/navsim-pilot-val1000-cache-config.yaml").write_text(OmegaConf.to_yaml(config))
        status("building", scenes=1000, cpu_workers=8, dataset="head_val only", scores_computed=False)
        worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=8)
        cache_data(config, worker)
        paths = sorted(CACHE.rglob("metric_cache.pkl"))
        assert len(paths) == 1000 and {p.parent.name for p in paths} == tokens
        metadata = CACHE / "metadata"
        metadata.mkdir(exist_ok=True)
        archive = CACHE / "metadata_original"
        archive.mkdir(exist_ok=True)
        for old in metadata.glob("*.csv"):
            shutil.move(str(old), str(archive / (old.name + "." + hashlib.sha256(old.read_bytes()).hexdigest()[:12])))
        with (metadata / "pilot-head-val1000.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file_name"])
            writer.writerows([[str(p)] for p in paths])
        loader = MetricCacheLoader(CACHE)
        assert set(loader.tokens) == tokens, f"Cache coverage mismatch: {len(loader.tokens)}"
        size = 0
        for token in sorted(tokens):
            path = Path(loader.metric_cache_paths[token])
            assert path.resolve().is_relative_to(CACHE.resolve())
            cached = loader.get_from_token(token)
            assert np.isfinite(cached.ego_state.rear_axle.serialize()).all()
            assert cached.trajectory is not None and cached.observation is not None
            size += path.stat().st_size
        report = {"cache_root": str(CACHE), "scenes": 1000, "requested_tokens": sorted(tokens),
                  "coverage": 1.0, "all_files_deserialized": True, "bytes": size,
                  "metadata_sha256": hashlib.sha256((metadata / "pilot-head-val1000.csv").read_bytes()).hexdigest(),
                  "tokens_sha256": hashlib.sha256((ROOT / "splits/pilot-head-val-tokens.txt").read_bytes()).hexdigest(),
                  "official_navsim_commit": "3e8291bfa89ff247231e0227778840cd0a036896",
                  "scores_computed": False, "navtest_used": False,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        (ROOT / "reports/navsim-pilot-val1000-cache.json").write_text(json.dumps(report, indent=2) + "\n")
        status("verified", scenes=1000, coverage=1.0, scores_computed=False)
    except Exception as error:
        status("failed", error=repr(error))
        raise
