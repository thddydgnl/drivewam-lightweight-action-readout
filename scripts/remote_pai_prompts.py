"""Precompute UMT5 prompt embeddings with the upstream `_encode_single_prompt` (batch one).

Stores only the valid rows (the upstream method zero-pads to 512); `PromptCache` restores the
identical [512, 4096] tensor. Prompt superset per clip: "", config default, every chunk value
in the clip's guidance store. Model environment, one GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "scripts"))
from pai_common import UP, config, prompt_key                      # noqa: E402
from modules.utils import load_text_encoder, load_tokenizer       # noqa: E402

def stores_for(name):
    if name == "final":
        data = json.loads((ROOT / "work/DriveWAM/src/physicalai/eval_data/prompts_test_sample_1k.json").read_text())
        clips = (ROOT / "splits/pai-final-1k.txt").read_text().split()
        return {c: data.get(c, {}) for c in clips}
    if name.startswith("train"):
        clips = (ROOT / f"splits/pai-{name.replace('train', 'train-')}.txt").read_text().split()
        base = ROOT / "data/physicalai/nav_prompts"
        out = {}
        for c in clips:
            f = base / f"{c}.json"
            out[c] = json.loads(f.read_text()).get("chunks", {}) if f.exists() else {}
        return out
    if name == "dev":
        data = json.loads((ROOT / "data/physicalai/dev_guidance/scene_evolving_guidance_store.json").read_text())
        clips = (ROOT / "splits/pai-dev-1k.txt").read_text().split()
        return {c: data.get(c, {}) for c in clips}
    raise ValueError(name)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--set", required=True); a = p.parse_args()
    cfg = config()
    out = ROOT / "caches/pai-prompts" / a.set
    out.mkdir(parents=True, exist_ok=True)
    stores = stores_for(a.set)
    prompts = {"", str(cfg.prompt)}
    for store in stores.values():
        prompts.update(str(v) for v in store.values())
    prompts = sorted(prompts)
    done = set()
    for meta in out.glob("*.safetensors.json"):
        done.update(json.loads(meta.read_text())["keys"])
    todo = [p_ for p_ in prompts if prompt_key(p_) not in done]
    print(json.dumps({"set": a.set, "clips": len(stores), "prompts": len(prompts), "todo": len(todo)}), flush=True)
    if todo:
        holder = UP["PhysicalAIEvaluator"].__new__(UP["PhysicalAIEvaluator"])
        holder.config, holder.dtype, holder.device = cfg, cfg.param_dtype, torch.device("cuda:0")
        holder.tokenizer = load_tokenizer(str(ROOT / "checkpoints/lingbot-va-base/tokenizer"))
        holder.text_encoder = load_text_encoder(str(ROOT / "checkpoints/lingbot-va-base/text_encoder"),
                                                torch_dtype=holder.dtype, torch_device=holder.device).eval()
        holder.text_encoder.requires_grad_(False)
        holder._prompt_cache = {}
        shard, keys, number = {}, [], len(list(out.glob("*.safetensors")))
        for i, prompt in enumerate(todo, 1):
            emb = holder._encode_single_prompt(prompt)                # upstream code path, [512, 4096]
            nonzero = emb.abs().sum(dim=1) != 0
            seq_len = int(nonzero.nonzero().max().item()) + 1 if nonzero.any() else 0
            assert torch.count_nonzero(emb[seq_len:]) == 0
            shard[prompt_key(prompt)] = emb[:seq_len].cpu().contiguous()
            keys.append(prompt_key(prompt))
            holder._prompt_cache.clear()
            if len(keys) == 2000 or i == len(todo):
                path = out / f"{number:05d}.safetensors"
                save_file(shard, str(path))
                Path(str(path) + ".json").write_text(json.dumps({"keys": keys, "set": a.set}) + "\n")
                number += 1; shard, keys = {}, []
                print(json.dumps({"set": a.set, "done": i, "total": len(todo)}), flush=True)
    print("verified", a.set, flush=True)
