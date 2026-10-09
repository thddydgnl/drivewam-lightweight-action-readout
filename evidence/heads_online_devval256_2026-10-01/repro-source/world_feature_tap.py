"""Observation-only current/first/terminal residual block feature extraction.

Hooks read selected video block outputs before norm_out/proj_out. They never
collect action block calls or change any block output. Full mode skips future
action denoising; current/first stop after their native video forward and clear
KV caches even when the original predict's cleanup has not been entered.
"""
import torch

PHASES = ["current_primer", "future_first_preupdate", "future_cache_terminal"]
LAYERS = [4, 9, 14, 19, 24, 29]

class FeatureStop(Exception):
    pass

class WorldFeatureTap:
    def __init__(self, policy, layers=LAYERS, collect_phases=None, capture_latents=True, record_metadata=True):
        self.policy, self.layers = policy, list(layers)
        self.collect_phases = list(PHASES if collect_phases is None else collect_phases)
        assert self.collect_phases and set(self.collect_phases) <= set(PHASES)
        self.capture_latents, self.record_metadata = capture_latents, record_metadata
        assert len(policy.transformer.blocks) == 30
        assert policy.config.action_chunk_count == 2 and len(policy.config.future_image_indices) == 4
        self.handles = [policy.transformer.register_forward_pre_hook(self.before, with_kwargs=True),
                        policy.transformer.register_forward_hook(self.after, with_kwargs=True)]
        for index in self.layers:
            self.handles.append(policy.transformer.blocks[index].register_forward_hook(
                lambda module, args, output, layer=index: self.block(layer, output)))
        self.enabled = False

    def reset(self, stop_at):
        self.stop_at, self.active_phase, self.video_calls, self.action_calls = stop_at, None, 0, 0
        self.features, self.phase_metadata, self.latents = {}, {}, {}
        self.skipped_future_action_calls = 0

    def before(self, module, args, kwargs):
        self.active_phase = None
        if not self.enabled:
            return
        if kwargs.get("action_mode", False):
            self.action_calls += 1
            return
        index = self.video_calls
        self.video_calls += 1
        inputs = args[0]
        phase = PHASES[0] if index == 0 else PHASES[1] if index == 1 else PHASES[2] if index == 4 else None
        if phase is not None:
            assert int(kwargs.get("update_cache", 0)) == {0:2,1:0,4:1}[index]
            self.active_phase = phase
            if phase in self.collect_phases:
                self.features[phase] = {}
            self.phase_metadata[phase] = {"input_timestep": inputs["timesteps"].detach().float().cpu().tolist() if self.record_metadata else None,
                "input_sigma": float(inputs["timesteps"].flatten()[0].item()/1000) if self.record_metadata else [0.,1.,None,None,5*.4/(1+4*.4)][index],
                "completed_video_updates": 0 if index in [0,1] else 3,
                "video_forward_index": index, "update_cache": int(kwargs.get("update_cache",0))}
            if self.capture_latents and phase in [PHASES[0], PHASES[2]]:
                self.latents[phase] = inputs["noisy_latents"].detach().clone()

    def block(self, layer, output):
        if not self.enabled or self.active_phase not in self.collect_phases:
            return
        assert isinstance(output, torch.Tensor) and output.shape == (1,112,3072)
        assert output.dtype == torch.bfloat16
        self.features[self.active_phase][layer] = output.detach().clone()

    def after(self, module, args, kwargs, output):
        if self.enabled and self.stop_at is not None and self.active_phase == self.stop_at:
            raise FeatureStop()

    def action_stub(self, chunk_index, action_shape, action_mask, prompt_emb, ego_cond, num_steps):
        assert chunk_index == 1 and action_shape[2] == 1
        self.skipped_future_action_calls += 1
        return torch.zeros(action_shape, device=self.policy.device, dtype=self.policy.dtype)

    def extract(self, batch, seed, stop_at=None, skip_actions=True, include_mean=True):
        assert stop_at is None or stop_at in PHASES
        self.reset(stop_at)
        original = self.policy._rollout_action_chunk
        if skip_actions:
            self.policy._rollout_action_chunk = self.action_stub
        self.enabled = True
        outputs = None
        try:
            with torch.inference_mode():
                outputs = self.policy.predict(batch, 3, 10, seed, video_denoise_fraction=.6)
        except FeatureStop:
            assert stop_at is not None
        finally:
            self.enabled = False
            self.policy._rollout_action_chunk = original
            self.policy.transformer.clear_cache(self.policy.cache_name)
        executed = PHASES if stop_at is None else PHASES[:PHASES.index(stop_at)+1]
        expected = [phase for phase in executed if phase in self.collect_phases]
        assert expected
        assert set(self.features) == set(expected)
        for phase in expected:
            assert set(self.features[phase]) == set(self.layers)
        if skip_actions:
            assert self.video_calls + self.action_calls == (6 if stop_at is None else [1,3,6][PHASES.index(stop_at)])
            assert self.skipped_future_action_calls == (1 if stop_at is None else 0)
        assert self.phase_metadata[PHASES[0]]["input_sigma"] == 0
        if PHASES[1] in expected:
            assert self.phase_metadata[PHASES[1]]["input_sigma"] == 1
        dense = torch.stack([torch.cat([self.features[phase][layer] for layer in self.layers],dim=0) for phase in expected])
        assert dense.shape == (len(expected),len(self.layers),112,3072)
        assert torch.isfinite(dense).all()
        # Use the same per-layer reduction for full cache and a single-layer
        # online probe, independent of how many phases/layers are requested.
        mean = torch.stack([torch.stack([layer.float().mean(dim=0) for layer in phase]) for phase in dense]) if include_mean else None
        latent = torch.cat([self.latents[phase] for phase in [PHASES[0],PHASES[2]] if phase in self.latents],dim=2) if self.capture_latents else None
        return {"dense":dense.detach(),"mean":mean.detach() if mean is not None else None,"latents":latent,
                "phase_metadata":dict(self.phase_metadata),"forward_calls":self.video_calls+self.action_calls,
                "original_outputs":outputs}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
