"""Drop-in replacement for DriveWAM's `VLLMNavCaptioner` using HF transformers.

vLLM wheels for Qwen3-VL require CUDA >= 12.8 drivers; the A5000 host has driver 560
(CUDA 12.6). This class keeps the upstream system prompt, message construction
(`_build_user_content`), image pixel bounds (256..512 x 28 x 28), chat template
(enable_thinking=False), greedy decoding (temperature 0) and `_clean` post-processing; only
the inference engine differs. Usage: `python pai_vlm_patch.py <guidance-script args>`.
"""
import sys
from pathlib import Path

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
sys.path.insert(0, str(ROOT / "work/DriveWAM/src"))

import torch                                                     # noqa: E402
import physicalai.vlm_preprocess.vlm_captioner as vc             # noqa: E402

def linear_patch_embed(model):
    """torch 2.9 + cuDNN 9.10: the bf16 Conv3d patch embedding (kernel == stride) takes ~10 s per
    call on the A5000. On pre-patchified input it equals one matmul, so run it as F.linear with
    the same weights (max deviation from an FP32 reference is identical: 1 bf16 ulp)."""
    pe = model.model.visual.patch_embed
    w, b = pe.proj.weight, pe.proj.bias
    def forward(hidden_states):
        x = hidden_states.to(w.dtype).view(-1, w[0].numel())
        return torch.nn.functional.linear(x, w.view(w.shape[0], -1), b).view(-1, pe.embed_dim)
    pe.forward = forward

class HFNavCaptioner(vc.VLLMNavCaptioner):
    def __init__(self, model_path, tensor_parallel_size=1, max_new_tokens=100, gpu_memory_utilization=0.9,
                 max_model_len=8192, max_images_per_prompt=3):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self._proc = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        self._proc.tokenizer.padding_side = "left"
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
        linear_patch_embed(self._model)
        self._max_new_tokens = max_new_tokens

    @torch.no_grad()
    def caption_batch(self, chunks):
        messages_list, texts, ids = [], [], []
        for ch in chunks:
            messages = [{"role": "system", "content": [{"type": "text", "text": vc.SYSTEM_PROMPT}]},
                        {"role": "user", "content": self._build_user_content(ch)}]
            messages_list.append(messages)
            texts.append(self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False))
            ids.append(ch["chunk_key"])
        images = []
        for messages in messages_list:
            image_inputs, _, _ = vc._process_vision(messages)
            images.extend(image_inputs or [])
        inputs = self._proc(text=texts, images=images or None, padding=True, return_tensors="pt").to("cuda:0")
        out = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        decoded = self._proc.batch_decode(new, skip_special_tokens=True)
        return {cid: vc._clean(text) for cid, text in zip(ids, decoded)}

vc.VLLMNavCaptioner = HFNavCaptioner

if __name__ == "__main__":
    import runpy
    sys.argv = ["generate_scene_evolving_guidance"] + sys.argv[1:]
    runpy.run_module("physicalai.vlm_preprocess.generate_scene_evolving_guidance", run_name="__main__")
