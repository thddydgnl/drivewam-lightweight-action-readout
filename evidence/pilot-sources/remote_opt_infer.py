"""Raw/opt 1/2/3/5/10 step validation and paired dev256 benchmark.

Uses one physical GPU, batch one, 20 warmups per model and 3 x 256 paired
measurements. No target file is opened here. CPU scoring runs after GPU work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from baseline_common import ROOT, file_sha, sample_seed, load_policy
from navsim_current_dataset import load_policy_input, build_rollout_batch

from navsim_action_opt import make_skip_terminal_action

MODELS = {"B10": 10, "B05": 5, "B10-opt": 10, "B05-opt": 5, "B03-opt": 3, "B02-opt": 2, "B01-opt": 1}
def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)

def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def telemetry():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,power.draw", "--format=csv,noheader"], text=True).strip()

class Instrumentation:
    def __init__(self, policy):
        self.policy, self.events, self.calls, self.originals = policy, [], [], {}
        phases = {"_encode_images": "vae_encode", "_encode_prompt": "prompt_cache",
                  "_prime_rollout_cache": "primer", "_rollout_video_frame": "world",
                  "_rollout_action_chunk": "action"}
        for method, phase in phases.items():
            original = getattr(policy, method)
            self.originals[method] = original
            def wrapper(*args, _original=original, _phase=phase, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = _original(*args, **kwargs)
                end.record()
                self.events.append((_phase, start, end))
                return result
            setattr(policy, method, wrapper)
        self.hook = policy.transformer.register_forward_pre_hook(self.forward, with_kwargs=True)
    def forward(self, module, args, kwargs):
        self.calls.append({"action_mode": bool(kwargs.get("action_mode", False)),
                           "update_cache": int(kwargs.get("update_cache", 0))})
    def reset(self):
        self.events.clear()
        self.calls.clear()
    def phases(self):
        result = {}
        for phase, start, end in self.events:
            result[phase + "_gpu_ms"] = result.get(phase + "_gpu_ms", 0.0) + start.elapsed_time(end)
        return result
    def close(self):
        self.hook.remove()
        for method, original in self.originals.items():
            setattr(self.policy, method, original)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="opt_devval256_2026-10-01")
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    assert not out.exists(), f"Refusing to overwrite an existing run: {out}"
    out.mkdir(parents=True)
    status_path = out / "inference-status.json"
    def status(stage, **extra):
        write_json(status_path, {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
        print(json.dumps({"stage": stage, **extra}), flush=True)
    try:
        torch.set_num_threads(8)
        assert torch.cuda.device_count() == 1, "Use LITE_GPU_IDS=0 so RNG/device and timing stay isolated"
        torch.cuda.set_device(0)
        tokens_path = ROOT / "splits/devval256-tokens.txt"
        tokens = tokens_path.read_text().splitlines()
        assert len(tokens) == len(set(tokens)) == 256
        manifest = ROOT / "splits/pilot-manifest.jsonl"
        data_lock = json.loads((ROOT / "DATA_LOCK.json").read_text())
        assert file_sha(manifest) == data_lock["manifest_sha256"]
        rows = {r["scene_token"]: r for r in (json.loads(line) for line in manifest.read_text().splitlines())}
        requested = [rows[token] for token in tokens]
        assert all(r["split"] == "head_val" and r["input_available"] and r["metric_cache_available"] for r in requested)
        source = ROOT / "work/DriveWAM"
        assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == "abbefb407c9c553f221522fc26077d307620adb4"
        assert subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True) == ""
        status("verifying_provenance")
        weight = ROOT / "checkpoints/drivewam/navsim/diffusion_pytorch_model.safetensors"
        weight_sha = file_sha(weight)
        assert weight_sha == "27e4846b0a25e22fc299f7aca90c01d802072d2b297741c85ed074b689ef26a9"
        status("preloading_observations")
        inputs = {}
        for row in requested:
            assert file_sha(row["input_path"]) == row["input_sha256"]
            inputs[row["scene_token"]] = load_policy_input(row["input_path"])
        status("loading_models")
        policy, loading_info = load_policy()
        raw_action, opt_action, opt_metadata = make_skip_terminal_action(policy)
        policy._skip_terminal_action_cache = False
        def dispatch(*args, **kwargs):
            method = opt_action if policy._skip_terminal_action_cache else raw_action
            return method(*args, **kwargs)
        policy._rollout_action_chunk = dispatch
        write_json(out / "action-opt-transformation.json", opt_metadata)
        smoke_tokens = (ROOT / "splits/smoke32-tokens.txt").read_text().splitlines()
        assert len(smoke_tokens) == len(set(smoke_tokens)) == 32
        for token in smoke_tokens:
            assert rows[token]["split"] == "head_train"
            assert file_sha(rows[token]["input_path"]) == rows[token]["input_sha256"]
            inputs[token] = load_policy_input(rows[token]["input_path"])
        previous = ROOT / "runs/baselines_devval256_2026-10-01"
        previous_protocol = json.loads((previous / "protocol.json").read_text())
        assert previous_protocol["tokens"] == tokens
        assert previous_protocol["checkpoint_sha256"] == weight_sha
        previous_predictions = {(r["model"], r["generation_seed"], r["scene_token"]): r
                                for r in (json.loads(line) for line in (previous / "prediction-index.jsonl").read_text().splitlines())}

        video_times = policy._prepare_video_timesteps(3, 0.6).tolist()
        schedules = {}
        for name, steps in MODELS.items():
            schedules[name] = {"action_timesteps": policy._prepare_timesteps(policy.action_scheduler, steps).tolist(),
                               "action_sigmas_solver": policy.action_scheduler.sigmas.tolist(),
                               "video_timesteps": video_times, "video_sigmas": policy.video_scheduler.sigmas.tolist()}
        protocol = {"run_id": args.run_id, "scope": "head-val dev256, not full head-val or navtest",
           "started_at_utc": datetime.now(timezone.utc).isoformat(), "tokens": tokens,
           "tokens_sha256": file_sha(tokens_path), "manifest_sha256": file_sha(manifest),
           "environment_lock_sha256": file_sha(ROOT / "ENVIRONMENT_LOCK.json"), "data_lock_sha256": file_sha(ROOT / "DATA_LOCK.json"),
           "checkpoint_sha256": weight_sha, "loading_info": loading_info,
           "model_source_commit": "abbefb407c9c553f221522fc26077d307620adb4", "source_modified": False,
           "batch_size": 1, "backend": "flashattn 2.8.3", "dtype": "bfloat16", "torch_compile": False,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "video_steps": 3, "video_denoise_fraction": 0.6, "action_steps": MODELS,
           "skip_terminal_action_cache": {name: name.endswith("-opt") for name in MODELS},
           "model_names": list(MODELS) + ["CV"], "gpu_model_names": list(MODELS),
           "expected_prediction_count": 2496, "cv_definition": "ego-local current vx/vy fixed for 4s; yaw fixed at zero; observation only",
           "previous_baseline_protocol_sha256": file_sha(previous / "protocol.json"),
           "parity_smoke_tokens": smoke_tokens, "parity_steps": [1, 2, 3, 5, 10], "schedules": schedules, "resolved_config": dict(policy.config),
           "primary_generation_seed": 0, "sample_seed": "big-endian first 4 SHA256 bytes of UTF-8 scene_token:generation_seed",
           "diagnostic_generation_seeds": [0, 1, 2], "diagnostic_tokens": tokens[:32],
           "warmup_per_model": 20, "timing_repeats": 3, "timing_samples_per_repeat": 256,
           "timing_scope": "preloaded CPU observation through batch construction, resize/H2D/VAE/primer/world/action/trajectory decoding and usable CPU trajectory; disk excluded",
           "phase_gpu_events": True, "forward_hook_counting": True, "other_gpu_jobs_and_cpu_scoring": "must be idle during timing",
           "telemetry_before": telemetry(), "scripts": {name: file_sha(ROOT / "scripts" / name)
                for name in ["baseline_common.py", "remote_baseline_infer.py", "remote_opt_infer.py", "navsim_action_opt.py", "navsim_current_dataset.py"]}}
        write_json(out / "protocol.json", protocol)
        instrument = Instrumentation(policy)
        resident = torch.cuda.memory_allocated(0)
        def run_one(token, model, generation_seed):
            policy._skip_terminal_action_cache = model.endswith("-opt")
            instrument.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            start = time.perf_counter()
            batch = build_rollout_batch(inputs[token], policy.config)
            with torch.inference_mode():
                actions, latents = policy.predict(batch, 3, MODELS[model], sample_seed(token, generation_seed), video_denoise_fraction=0.6)
                trajectory = policy.pred_traj_to_poses(policy.actions_to_trajectory(actions))
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            assert len(instrument.calls) == (MODELS[model] + 6 + (not model.endswith("-opt"))), instrument.calls
            if model.endswith("-opt"):
                assert not any(c["action_mode"] and c["update_cache"] == 1 for c in instrument.calls[2:])
            assert trajectory.shape == (8, 3) and np.isfinite(trajectory).all()
            assert torch.isfinite(actions).all() and torch.isfinite(latents).all()
            timing = {"model": model, "scene_token": token, "generation_seed": generation_seed,
                      "sample_seed": sample_seed(token, generation_seed), "policy_ms": elapsed,
                      "actual_forward_calls": len(instrument.calls), "calls": list(instrument.calls),
                      "skip_terminal_action_cache": model.endswith("-opt"),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), **instrument.phases()}
            return trajectory, actions, latents, timing
        status("raw_opt_parity", completed=0, total=160)
        parity_rows = []
        for number, token in enumerate(smoke_tokens):
            for steps in [1, 2, 3, 5, 10]:
                outputs = []
                for skip in [False, True]:
                    policy._skip_terminal_action_cache = skip
                    instrument.reset()
                    with torch.inference_mode():
                        batch = build_rollout_batch(inputs[token], policy.config)
                        actions, latents = policy.predict(batch, 3, steps, sample_seed(token, 0), video_denoise_fraction=0.6)
                        trajectory = policy.pred_traj_to_poses(policy.actions_to_trajectory(actions))
                    torch.cuda.synchronize()
                    assert len(instrument.calls) == steps + 6 + (not skip)
                    outputs.append((tensor_sha(actions), tensor_sha(latents), hashlib.sha256(trajectory.tobytes()).hexdigest()))
                    del actions, latents
                assert outputs[0] == outputs[1], (token, steps, "raw/opt differs")
                parity_rows.append({"scene_token": token, "steps": steps, "generation_seed": 0,
                                    "action_sha256": outputs[0][0], "world_sha256": outputs[0][1],
                                    "trajectory_sha256": outputs[0][2], "bitwise_equal": True})
            if (number + 1) % 8 == 0:
                status("raw_opt_parity", completed=(number + 1)*5, total=160)
        write_json(out / "raw-opt-parity.json", {"scenes": 32, "comparisons": 160, "all_bitwise_equal": True, "rows": parity_rows})
        # Verify that phase wrappers/count hooks do not alter the original path.
        status("instrumentation_parity")
        checks = {}
        for model, steps in MODELS.items():
            tok = tokens[0]
            traj, actions, latents, _ = run_one(tok, model, 0)
            hashes = (tensor_sha(actions), tensor_sha(latents), hashlib.sha256(traj.tobytes()).hexdigest())
            instrument.close()
            with torch.inference_mode():
                batch = build_rollout_batch(inputs[tok], policy.config)
                a, z = policy.predict(batch, 3, steps, sample_seed(tok, 0), video_denoise_fraction=0.6)
                t = policy.pred_traj_to_poses(policy.actions_to_trajectory(a))
            assert hashes == (tensor_sha(a), tensor_sha(z), hashlib.sha256(t.tobytes()).hexdigest())
            checks[model] = True
            del actions, latents, a, z
            instrument = Instrumentation(policy)
        write_json(out / "instrumentation-parity.json", checks)
        status("warmup", per_model=20)
        for token in tokens[:20]:
            for model in MODELS:
                trajectory, actions, latents, _ = run_one(token, model, 0)
                del actions, latents
        write_json(out / "warmup-telemetry.json", {"resident_allocated_bytes": resident, "after_warmup": telemetry()})
        hashes_by_key = {}
        timings_path = out / "timings.jsonl"
        predictions_path = out / "prediction-index.jsonl"
        with timings_path.open("w") as timings, predictions_path.open("w") as predictions:
            def pair(token, generation_seed, repeat, save_predictions):
                world_hashes = []
                order = list(MODELS)
                offset = (tokens.index(token) + repeat) % len(order)
                order = order[offset:] + order[:offset]
                if (tokens.index(token) + repeat) % 2:
                    order.reverse()
                for model in order:
                    trajectory, actions, latents, timing = run_one(token, model, generation_seed)
                    action_hash, world_hash = tensor_sha(actions), tensor_sha(latents)
                    world_hashes.append(world_hash)
                    if model in ["B10", "B05", "B10-opt", "B05-opt"]:
                        previous_key = (model.replace("-opt", ""), generation_seed, token)
                        reference = previous_predictions[previous_key]
                        assert world_hash == reference["world_latents_sha256"]
                        assert hashlib.sha256(trajectory.tobytes()).hexdigest() == reference["trajectory_sha256"], "Prior raw baseline or full dev256 raw/opt parity failed"
                        if not model.endswith("-opt"):
                            assert file_sha(reference["prediction_path"]) == reference["prediction_sha256"]
                    key = (token, model, generation_seed)
                    digest = hashlib.sha256(trajectory.tobytes()).hexdigest()
                    if key in hashes_by_key:
                        assert hashes_by_key[key] == (action_hash, world_hash, digest), "Fixed seed repeated prediction changed"
                    else:
                        hashes_by_key[key] = action_hash, world_hash, digest
                    timing.update(repeat=repeat, log_name=rows[token]["log_name"], trajectory_sha256=digest,
                                  world_latents_sha256=world_hash, action_tensor_sha256=action_hash)
                    timings.write(json.dumps(timing) + "\n")
                    timings.flush()
                    if save_predictions:
                        folder = out / "predictions" / model / f"generation_seed_{generation_seed}"
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / (token + ".npz")
                        np.savez(path, trajectory=trajectory, actions=actions.float().cpu().numpy(), scene_token=np.asarray(token))
                        record = {k: timing[k] for k in ["model", "scene_token", "generation_seed", "sample_seed", "log_name", "actual_forward_calls", "trajectory_sha256", "world_latents_sha256"]}
                        record.update(prediction_path=str(path), prediction_sha256=file_sha(path))
                        predictions.write(json.dumps(record) + "\n")
                        predictions.flush()
                    del actions, latents
                assert len(set(world_hashes)) == 1, "Raw/opt candidate generated worlds differ"
            for repeat in range(3):
                status("primary_and_online_timing", repeat=repeat, completed=0, total=256)
                for number, token in enumerate(tokens):
                    pair(token, 0, repeat, repeat == 0)
                    if (number + 1) % 16 == 0:
                        status("primary_and_online_timing", repeat=repeat, completed=number + 1, total=256)
                write_json(out / f"telemetry-repeat-{repeat}.json", {"telemetry": telemetry()})
            for generation_seed in [1, 2]:
                status("generation_seed_diagnostic", generation_seed=generation_seed, completed=0, total=32)
                for number, token in enumerate(tokens[:32]):
                    pair(token, generation_seed, 0, True)
                    if (number + 1) % 16 == 0:
                        status("generation_seed_diagnostic", generation_seed=generation_seed, completed=number + 1, total=32)
        instrument.close()
        # Observation-only constant velocity reference, CPU wall time; separate
        # from GPU policy measurements and not a B* candidate.
        with timings_path.open("a") as timings, predictions_path.open("a") as predictions:
            def cv(token):
                start = time.perf_counter()
                velocity = inputs[token]["ego_velocity_acc"][:2].numpy()
                trajectory = np.zeros((8, 3), dtype=np.float32)
                trajectory[:, :2] = np.arange(1, 9, dtype=np.float32)[:, None] * .5 * velocity
                return trajectory, (time.perf_counter() - start) * 1000
            for token in tokens[:20]:
                cv(token)
            for repeat in range(3):
                for token in tokens:
                    trajectory, elapsed = cv(token)
                    digest = hashlib.sha256(trajectory.tobytes()).hexdigest()
                    record = {"model": "CV", "scene_token": token, "generation_seed": 0,
                              "sample_seed": sample_seed(token, 0), "log_name": rows[token]["log_name"],
                              "actual_forward_calls": 0, "trajectory_sha256": digest, "world_latents_sha256": "not_applicable"}
                    timings.write(json.dumps({**record, "repeat": repeat, "policy_ms": elapsed,
                                   "execution_device": "cpu", "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}) + "\n")
                    if repeat == 0:
                        folder = out / "predictions/CV/generation_seed_0"
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / (token + ".npz")
                        np.savez(path, trajectory=trajectory, scene_token=np.asarray(token))
                        record.update(prediction_path=str(path), prediction_sha256=file_sha(path))
                        predictions.write(json.dumps(record) + "\n")
        records = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        assert len(records) == 2496
        write_json(out / "inference-summary.json", {"primary_predictions": 2048, "diagnostic_extra_predictions": 448,
            "all_outputs_finite": True, "requested_primary_tokens": 256, "primary_coverage_each_model": 1.0,
            "paired_world_bitwise_equal": True, "three_repeat_prediction_bitwise_equal": True,
            "instrumentation_parity": checks, "timing_samples_each_model": 768,
            "raw_opt_smoke_comparisons": 160, "B10_B05_full_dev256_prior_prediction_parity": True,
            "predictions_sha256": file_sha(predictions_path), "timings_sha256": file_sha(timings_path),
            "telemetry_after": telemetry(), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        status("verified", primary_predictions=2048, diagnostic_extra_predictions=448)
    except Exception as exc:
        status("failed", error=repr(exc))
        raise
