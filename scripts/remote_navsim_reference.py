"""Prepare one original processor fixture and verify current-only parity."""
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import runpy
import sys

import numpy as np
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
if __name__ == "__main__":
    token = (ROOT / "splits/smoke32-tokens.txt").read_text().splitlines()[0]
    row = next(json.loads(line) for line in (ROOT / "splits/pilot-manifest.jsonl").read_text().splitlines()
               if json.loads(line)["scene_token"] == token)
    sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))
    module = runpy.run_path(str(ROOT / "work/DriveWAM/src/navsim/process_data.py"), run_name="navsim_reference_processor")
    Processor = module["NavSimDataProcessor"]
    processor = Processor.__new__(Processor)
    processor.num_history_frames = 4
    processor.num_future_frames = 10
    sensors = SensorConfig.build_no_sensors()
    sensors.cam_f0 = True
    processor.scene_loader = SceneLoader(ROOT / "data/navsim_logs/trainval", ROOT / "data/sensor_blobs/trainval",
                SceneFilter(4, 10, 1, True, log_names=[row["log_name"]], tokens=[token]), sensors)
    reference = processor.process_scene(token)
    with np.load(row["input_path"], allow_pickle=False) as obs, np.load(row["label_path"], allow_pickle=False) as target:
        np.testing.assert_array_equal(obs["rgb_current"], reference["images"][0])
        np.testing.assert_allclose(obs["history_poses"], reference["history_poses"], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(obs["ego_velocity_acc"], np.concatenate([
                           reference["ego_status"]["velocity"], reference["ego_status"]["acceleration"]]))
        assert int(obs["command_id"]) == int(np.argmax(reference["ego_status"]["driving_command"]))
        np.testing.assert_allclose(target["future_poses_5s"], np.stack([
                           step["pose"] for step in reference["future_trajectory"]]), rtol=0, atol=1e-5)
    dest = ROOT / "processed/reference_smoke"
    dest.mkdir(exist_ok=True)
    path = dest / ("sample_" + token + ".pkl")
    with path.open("wb") as output:
        pickle.dump(reference, output, protocol=pickle.HIGHEST_PROTOCOL)
    report = {"scene_token": token, "log_name": row["log_name"], "fixture_path": str(path),
              "original_processor_source_modified": False, "original_current_rgb_exact_parity": True,
              "history_ego_command_parity": True, "future_label_parity": True,
              "fixture_contains_future_rgb": True, "fixture_use": "reference parity only; excluded from policy training inputs",
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / "reports/navsim-reference-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
