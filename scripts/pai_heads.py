"""PhysicalAI student head (pre-registration §5): SF3 trunk + FP32 40-step regression output.

Same trunk as the NAVSIM v3 head (query pooling over layers [10,20,30], width 512, 16 queries)
with the PhysicalAI ego condition (vel/acc 4 + curvature 1) and quantile normalization.
Decoding matches upstream `actions_to_trajectory_with_reset` within a chunk:
pose = cumsum((n + 1) / 2 * (q99 - q01 + 1e-6) + q01). Dataset actions are deltas in the clip-start
frame; the head predicts in the chunk-start ego frame and `to_clip_frame` rotates by the heading
accumulated from the registered GT actions (pre-registration change 9; `frame="clip"` = original spec).
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lightweight_heads import QueryPooler

Q01 = [-0.50788880, -1.31925802, -0.03548078]
Q99 = [3.50935863, 1.24382361, 0.03866158]
STEPS = 40

def decode(normalized):
    n = normalized.float()
    lo, hi = n.new_tensor(Q01), n.new_tensor(Q99)
    return torch.cumsum((n + 1.0) / 2.0 * (hi - lo + 1e-6) + lo, dim=-2)

def denorm(n):
    lo, hi = n.new_tensor(Q01), n.new_tensor(Q99)
    return (n + 1.0) / 2.0 * (hi - lo + 1e-6) + lo

def renorm(d):
    lo, hi = d.new_tensor(Q01), d.new_tensor(Q99)
    return (d - lo) / (hi - lo + 1e-6) * 2.0 - 1.0

def heading_before(prev_norm_chunks):
    """Heading of a chunk start relative to the clip start, from the registered GT action chunks
    before it ([k-1, 40, 3] normalized, clip frame). Same float64 arithmetic online and offline."""
    total = 0.0
    for chunk in prev_norm_chunks:
        d = (np.asarray(chunk, dtype=np.float64)[:, 2] + 1.0) / 2.0 * (Q99[2] - Q01[2] + 1e-6) + Q01[2]
        total += float(np.sum(d))
    return total

def to_clip_frame(norm_ego, theta):
    """Head output = normalized deltas in the chunk-start ego frame -> normalized clip-frame deltas
    (the dataset/metric frame), rotating x/y by the chunk-start heading theta [B]."""
    d = denorm(norm_ego.float())
    c, s = torch.cos(theta).float()[:, None], torch.sin(theta).float()[:, None]
    x, y = c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]
    return renorm(torch.stack([x, y, d[..., 2]], dim=-1))

def loss_fn(normalized, target_normalized):
    pred, gt = decode(normalized), decode(target_normalized)
    position = F.huber_loss(pred[..., :2], gt[..., :2], delta=1.0)
    yaw = (1 - torch.cos(pred[..., 2] - gt[..., 2])).mean()
    delta = F.huber_loss(normalized.float(), target_normalized.float(), delta=1.0)
    return position + 0.2 * yaw + 0.1 * delta

class PaiHead(nn.Module):
    def __init__(self, layers=3, width=512, queries=16, ego_dim=5):
        super().__init__()
        self.poolers = nn.ModuleList(QueryPooler(width, queries) for _ in range(layers))
        self.ego = nn.Sequential(nn.Linear(ego_dim, 128), nn.GELU(), nn.Linear(128, 128))
        self.fuse = nn.Sequential(nn.Linear(layers * width + 128, 512), nn.GELU())
        self.residual = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512))
        self.time = nn.Parameter(torch.randn(STEPS, 512) * 0.02)
        self.out = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 3))
    def forward(self, world, ego):
        state = torch.cat([p(world[:, i]) for i, p in enumerate(self.poolers)] + [self.ego(ego)], dim=-1)
        state = self.fuse(state)
        state = state + self.residual(state)
        with torch.autocast("cuda", enabled=False):
            return self.out(state.float()[:, None, :] + self.time[None])            # [B, 40, 3] normalized deltas
