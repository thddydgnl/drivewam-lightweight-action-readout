"""Batch-one online head policy with native current/first/terminal exits."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from lightweight_heads import build_head, decode_poses
from navsim_current_dataset import build_rollout_batch
from world_feature_tap import WorldFeatureTap, PHASES

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def observation_ego(observation):
    return torch.cat([observation["history_poses"].flatten(),observation["ego_velocity_acc"],
                      torch.nn.functional.one_hot(observation["command_id"],num_classes=4).float()])

class OnlineHeadPolicy:
    def __init__(self,folder,world_policy=None,device="cuda:0",cache_only=False):
        self.folder=Path(folder);self.summary=json.loads((self.folder/"summary.json").read_text())
        self.config=self.summary["config"];self.kind=self.config["kind"];self.device=torch.device(device)
        group="fusion" if self.kind in ["sf1","sf1-wide","sf3"] else "probes"
        root=self.folder.parent;protocol=json.loads((root/(group+"-protocol.json")).read_text())
        assert sha(root/(group+"-protocol.json"))==self.summary["protocol_sha256"]
        stats_path=root/(group+"-stats.safetensors")
        assert sha(stats_path)==protocol["stats_sha256"]
        self.stats=load_file(str(stats_path),device="cpu")
        checkpoint=self.summary["best_by_pilot_PDMS"]["checkpoint"]
        assert Path(checkpoint).resolve().is_relative_to(self.folder.resolve())
        assert sha(checkpoint)==self.summary["best_by_pilot_PDMS"]["checkpoint_sha256"]
        self.model=build_head(self.kind)
        self.model.load_state_dict(load_file(checkpoint,device="cpu"),strict=True)
        self.model=self.model.to(self.device).eval().requires_grad_(False)
        self.world_policy=world_policy;self.tap=None
        self.record_head_events=False;self.head_events=None
        self.phase={"current":0,"first":1,"terminal":2}[self.config["phase"]]
        if self.kind!="ego" and not cache_only:
            assert world_policy is not None
            self.layers=[9,19,29] if self.kind=="sf3" else [29]
            self.tap=WorldFeatureTap(world_policy,self.layers,collect_phases=[PHASES[self.phase]],
                                     capture_latents=False,record_metadata=False)

    def predict(self,observation,seed):
        ego=(observation_ego(observation)-self.stats["ego_mean"])/self.stats["ego_std"]
        ego=ego[None].to(self.device)
        with torch.inference_mode():
            if self.kind=="ego":
                world=torch.empty((1,3072),device=self.device);calls=0
            else:
                assert self.tap is not None, "A cache-only policy cannot run online"
                batch=build_rollout_batch(observation,self.world_policy.config)
                result=self.tap.extract(batch,seed,stop_at=PHASES[self.phase],
                                        include_mean=self.kind in ["linear","fmlp"])
                calls=result["forward_calls"]
                if self.kind in ["linear","fmlp"]:
                    world=result["mean"][0]
                    if self.kind=="linear":
                        world=((world.cpu()-self.stats["world_mean"][self.phase])/self.stats["world_std"][self.phase]).to(self.device)
                else:world=result["dense"][0][None]
            # Backbone retains its original precision path. Autocast applies
            # only to the small head, exactly as in offline head evaluation.
            if self.record_head_events:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
            with torch.autocast("cuda",dtype=torch.bfloat16):normalized=self.model(world,ego)
            poses=decode_poses(normalized)[0]
            if self.record_head_events:
                end.record();self.head_events=(start,end)
            trajectory=poses.cpu().numpy()
        assert trajectory.shape==(8,3)
        return trajectory,calls

    def predict_cached(self,mean,dense,ego):
        ego=((ego-self.stats["ego_mean"])/self.stats["ego_std"])[None].to(self.device)
        with torch.inference_mode():
            if self.kind in ["sf1","sf1-wide","sf3"]:
                world=dense[None].to(self.device)
                if self.kind in ["sf1","sf1-wide"]:world=world[:,-1:]
            else:
                world=mean[None]
                if self.kind=="linear":world=(world-self.stats["world_mean"][self.phase])/self.stats["world_std"][self.phase]
                world=world.to(self.device)
            with torch.autocast("cuda",dtype=torch.bfloat16):normalized=self.model(world,ego)
            return decode_poses(normalized)[0].cpu().numpy()

    def close(self):
        if self.tap:self.tap.close()
        self.model.cpu()
