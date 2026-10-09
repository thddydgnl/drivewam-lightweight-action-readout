"""Read prepared NAVSIM PolicyInput files without opening any target file.

Run inside the model environment with the pinned DriveWAM config. The original
rollout needs five image slots, but reads pixels only from the current slot;
unused future slots are zero placeholders sized from config.
"""
import json
from pathlib import Path

import numpy as np
import torch

INPUT_KEYS = {"rgb_current", "history_poses", "ego_velocity_acc", "command_id"}

def load_policy_input(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == INPUT_KEYS, "Unexpected fields in PolicyInput"
        values = {key: stored[key].copy() for key in INPUT_KEYS}
    assert values["rgb_current"].dtype == np.uint8
    assert values["rgb_current"].ndim == 3 and values["rgb_current"].shape[-1] == 3
    assert values["history_poses"].shape == (4, 3)
    assert values["ego_velocity_acc"].shape == (4,)
    assert all(np.isfinite(value).all() for value in values.values())
    return {"rgb_current": torch.from_numpy(values["rgb_current"]).permute(2, 0, 1).contiguous(),
            "history_poses": torch.from_numpy(values["history_poses"]),
            "ego_velocity_acc": torch.from_numpy(values["ego_velocity_acc"]),
            "command_id": torch.tensor(int(values["command_id"]), dtype=torch.long),
            "scene_token": path.stem, "input_path": str(path)}

def build_rollout_batch(policy_input, config):
    assert config.action_chunk_count == 2 and config.action_chunk_steps == 8
    assert config.history_num_frames == 4 and len(config.future_image_indices) == 4
    current = policy_input["rgb_current"]
    images = torch.zeros(1, 1 + len(config.future_image_indices), *current.shape, dtype=torch.uint8)
    images[0, 0] = current
    shape = (1, config.action_dim, config.action_chunk_count, config.action_chunk_steps, 1)
    actions = torch.zeros(shape, dtype=torch.float32)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[:, list(config.used_action_channel_ids)[:3], 1] = True
    command = int(policy_input["command_id"])
    assert 0 <= command < len(config.command_prompts)
    return {"images": images, "prompt": [config.command_prompts[command]],
            "command_id": policy_input["command_id"].reshape(1),
            "history_poses": policy_input["history_poses"].unsqueeze(0),
            "ego_velocity_acc": policy_input["ego_velocity_acc"].unsqueeze(0),
            "actions": actions, "actions_mask": mask,
            "scene_token": [policy_input["scene_token"]]}

class NavsimCurrentDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split):
        self.rows = [row for row in (json.loads(line) for line in Path(manifest).read_text().splitlines())
                     if row["split"] == split]
        assert self.rows and all(row["input_available"] for row in self.rows)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        value = load_policy_input(row["input_path"])
        assert value["scene_token"] == row["scene_token"]
        return value

def load_target(path):
    """Explicit training/scoring API; never called by PolicyInput or rollout."""
    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == {"future_poses_5s", "future_poses_4s", "action_deltas_4s"}
        return {key: torch.from_numpy(stored[key].copy()) for key in stored.files}
