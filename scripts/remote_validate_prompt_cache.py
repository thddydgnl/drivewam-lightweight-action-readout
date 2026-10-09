"""Validate the serialized four-prompt cache and record its provenance."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))
from configs import VA_CONFIGS
path = ROOT / "caches/prompt/navsim-four-prompts.pt"
cache = torch.load(path, map_location="cpu", weights_only=True)
prompts = VA_CONFIGS["navsim_cfg"].command_prompts
assert list(cache) == prompts
assert all(t.shape == (512, 4096) and t.dtype == torch.bfloat16 and torch.isfinite(t).all()
           for t in cache.values())
metadata = {"cache_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "source_commit": "abbefb407c9c553f221522fc26077d307620adb4",
    "encoder_and_tokenizer_repo": "robbyant/lingbot-va-base",
    "encoder_and_tokenizer_revision": "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c",
    "encoding_method": "Pinned NavSimEvaluator._encode_prompt, eval mode, BF16",
    "torch": torch.__version__, "serialized_cache_readback": "passed",
    "commands": [{"command_id": i, "prompt": prompt, "shape": [512, 4096],
                  "tensor_sha256": hashlib.sha256(cache[prompt].contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()}
                 for i, prompt in enumerate(prompts)]}
expected = json.loads((ROOT / "reports/checkpoint-smoke.json").read_text())["prompt_cache"]["sha256"]
assert metadata["sha256"] == expected
(ROOT / "caches/prompt/navsim-four-prompts.json").write_text(json.dumps(metadata, indent=2) + "\n")
(ROOT / "reports/prompt-cache-readback.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps({"serialized_cache_readback": "passed", "commands": len(prompts)}, indent=2))
