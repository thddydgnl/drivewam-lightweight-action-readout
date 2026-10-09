"""Round-3 heads (pre-registration §6): SF3 trunk with FP32 output layers.

R: regression of normalized deltas (v2 decoder, run in FP32).
A: anchor head = classification over K k-means trajectories + per-anchor offset.
"""
import torch
from torch import nn
import torch.nn.functional as F
from lightweight_heads import QueryPooler, EgoEncoder, TrajectoryDecoder, decode_poses, action_loss

class SF3Trunk(nn.Module):
    def __init__(self, layers=3, width=512, queries=16):
        super().__init__()
        self.layers = layers
        self.poolers = nn.ModuleList(QueryPooler(width, queries) for _ in range(layers))
        self.ego = EgoEncoder()
        self.fuse = nn.Sequential(nn.Linear(layers * width + 128, 512), nn.GELU())
        self.residual = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512))
    def forward(self, world, ego):
        assert world.ndim == 4 and world.shape[1] == self.layers
        state = torch.cat([p(world[:, i]) for i, p in enumerate(self.poolers)] + [self.ego(ego)], dim=-1)
        state = self.fuse(state)
        return state + self.residual(state)

class RegressionHead(nn.Module):
    kind = "R"
    def __init__(self):
        super().__init__()
        self.trunk, self.decoder = SF3Trunk(), TrajectoryDecoder()
    def forward(self, world, ego):
        state = self.trunk(world, ego)
        with torch.autocast("cuda", enabled=False):
            normalized = self.decoder(state.float())
        return {"normalized": normalized, "poses": decode_poses(normalized)}
    def loss(self, out, poses, target, anchor_index=None):
        return action_loss(out["normalized"], poses, target)[0]

class AnchorHead(nn.Module):
    kind = "A"
    def __init__(self, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors.float())          # [K, 8, 3]
        k = anchors.shape[0]
        self.trunk = SF3Trunk()
        self.norm = nn.LayerNorm(512)
        self.logits = nn.Linear(512, k)
        self.offsets = nn.Linear(512, k * 24)
        nn.init.zeros_(self.offsets.weight); nn.init.zeros_(self.offsets.bias)
    def forward(self, world, ego):
        state = self.trunk(world, ego)
        with torch.autocast("cuda", enabled=False):
            h = self.norm(state.float())
            logits = self.logits(h)
            offsets = self.offsets(h).view(len(h), -1, 8, 3)
            candidates = self.anchors[None] + offsets                  # [B, K, 8, 3]
            best = logits.argmax(-1)
            poses = candidates[torch.arange(len(h), device=h.device), best]
        return {"logits": logits, "candidates": candidates, "poses": poses}
    def loss(self, out, poses, target, anchor_index):
        chosen = out["candidates"][torch.arange(len(poses), device=poses.device), anchor_index]
        position = F.huber_loss(chosen[..., :2], poses[..., :2], delta=1.)
        yaw = (1 - torch.cos(chosen[..., 2] - poses[..., 2])).mean()
        return F.cross_entropy(out["logits"], anchor_index) + position + .2 * yaw

def kmeans_anchors(poses, k=256, iterations=50, seed=17):
    """Lloyd k-means on x/y of 8 waypoints; heading = circular mean of members."""
    xy = poses[..., :2].reshape(len(poses), -1).double()
    g = torch.Generator().manual_seed(seed)
    centers = xy[torch.randint(len(xy), (1,), generator=g)]
    for _ in range(1, k):                                              # k-means++
        d = torch.cdist(xy, centers).min(1).values ** 2
        centers = torch.cat([centers, xy[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for _ in range(iterations):
        assign = torch.cdist(xy, centers).argmin(1)
        for j in range(k):
            members = xy[assign == j]
            if len(members):
                centers[j] = members.mean(0)
    assign = torch.cdist(xy, centers).argmin(1)
    heading = torch.zeros(k, 8, dtype=torch.float64)
    for j in range(k):
        h = poses[assign == j][..., 2].double()
        if len(h):
            heading[j] = torch.atan2(torch.sin(h).mean(0), torch.cos(h).mean(0))
    return torch.cat([centers.view(k, 8, 2), heading[..., None]], -1).float()

def nearest_anchor(anchors, poses):
    """Index of the anchor with the smallest mean x/y displacement to each target."""
    d = torch.linalg.vector_norm(anchors[None, :, :, :2] - poses[:, None, :, :2], dim=-1).mean(-1)
    return d.argmin(1)
