"""Verify isolated NAVSIM v1.1 and nuPlan imports and the CPU PDM simulator."""
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from threadpoolctl import threadpool_info
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
identity_error = float(np.abs(np.linalg.pinv(np.eye(40)) - np.eye(40)).max())
assert identity_error < 1e-12, f"BLAS numeric failure: pinv(I) error={identity_error}"
sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
ego = EgoState.build_from_rear_axle(StateSE2(0.0, 0.0, 0.0), StateVector2D(5.0, 0.0),
           StateVector2D(0.0, 0.0), 0.0, TimePoint(0), get_pacifica_parameters())
states = np.tile(ego_state_to_state_array(ego), (1, 41, 1))
states[0, :, StateIndex.X] = np.arange(41) * 0.5
simulated = PDMSimulator(sampling).simulate_proposals(states, ego)
assert simulated.shape == states.shape and np.isfinite(simulated).all()
assert abs(simulated[0, -1, StateIndex.X] - 20.0) < 0.5
scorer = PDMScorer(sampling)
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "packages": {p: importlib.metadata.version(p) for p in
                       ("navsim", "nuplan-devkit", "torch", "numpy", "Shapely", "hydra-core")},
          "torch_cuda": torch.version.cuda, "scorer_imports": "passed",
          "blas": threadpool_info(), "pinv_identity_max_abs_error": identity_error,
          "simulator": {"shape": list(simulated.shape),
                        "straight_4s_final_x": float(simulated[0, -1, StateIndex.X])},
          "pdms_on_real_scene": "not_run"}
(ROOT / "reports/scorer-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
