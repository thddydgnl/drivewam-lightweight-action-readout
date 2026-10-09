"""Small independent action readouts of frozen, observation-derived features.

Architecture follows the project's research protocol. No upstream runtime is
imported or trained. Query pooling/fusion is independently implemented here.
"""
import torch
from torch import nn
import torch.nn.functional as F

Q01 = [-.01582209,-1.76743735,-.18790549]
Q99 = [6.15755262,2.46998216,.18174374]

def normalize_deltas(deltas):
    lo,hi=deltas.new_tensor(Q01),deltas.new_tensor(Q99)
    return (deltas-lo)/(hi-lo)*2-1

def decode_poses(normalized):
    normalized=normalized.float()
    lo,hi=normalized.new_tensor(Q01),normalized.new_tensor(Q99)
    return torch.cumsum((normalized+1)*.5*(hi-lo)+lo,dim=-2)

def action_loss(normalized,poses,normalized_target):
    predicted=decode_poses(normalized)
    position=F.huber_loss(predicted[...,:2],poses[...,:2],delta=1.)
    yaw=(1-torch.cos(predicted[...,2]-poses[...,2])).mean()
    delta=F.huber_loss(normalized.float(),normalized_target,delta=1.)
    return position+.2*yaw+.1*delta,{"position":position.detach(),"yaw":yaw.detach(),"delta":delta.detach()}

class EgoEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(20,128),nn.GELU(),nn.Linear(128,128))

class TrajectoryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=nn.Parameter(torch.randn(8,512)*.02)
        self.out=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Linear(256,3))
    def forward(self,state):
        return self.out(state[:,None,:]+self.time[None])

class LinearReadout(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(3072+20,24)
    def forward(self,world,ego):
        return self.linear(torch.cat([world,ego],dim=-1)).reshape(-1,8,3)

class MeanMLPReadout(nn.Module):
    def __init__(self,ego_only=False):
        super().__init__();self.ego_only=ego_only;self.ego=EgoEncoder()
        if not ego_only:
            self.world=nn.Sequential(nn.LayerNorm(3072),nn.Linear(3072,512),nn.GELU(),nn.Linear(512,512))
        self.fuse=nn.Sequential(nn.Linear(128 if ego_only else 640,512),nn.GELU(),nn.Linear(512,512))
        self.decoder=TrajectoryDecoder()
    def forward(self,world,ego):
        state=self.ego(ego)
        if not self.ego_only:state=torch.cat([self.world(world),state],dim=-1)
        return self.decoder(self.fuse(state))

class QueryPooler(nn.Module):
    def __init__(self,width=512,queries=16):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(3072),nn.Linear(3072,width))
        self.query=nn.Parameter(torch.randn(queries,width)*.02)
        self.attention=nn.MultiheadAttention(width,8,batch_first=True,dropout=0.)
        self.merge=nn.Parameter(torch.zeros(queries))
        self.norm=nn.LayerNorm(width)
    def forward(self,tokens):
        projected=self.project(tokens)
        query=self.query[None].expand(len(tokens),-1,-1)
        attended,_=self.attention(query,projected,projected,need_weights=False)
        state=(attended*self.merge.softmax(dim=0)[None,:,None]).sum(dim=1)
        return self.norm(state)

class StateFusionReadout(nn.Module):
    def __init__(self,layers=3,width=512,queries=16):
        super().__init__();self.layers=layers;self.width=width
        self.poolers=nn.ModuleList(QueryPooler(width,queries) for _ in range(layers))
        self.ego=EgoEncoder()
        self.fuse=nn.Sequential(nn.Linear(layers*width+128,512),nn.GELU())
        self.residual=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,1024),nn.GELU(),nn.Linear(1024,512))
        self.decoder=TrajectoryDecoder()
    def forward(self,world,ego):
        assert world.ndim==4 and world.shape[1]==self.layers
        state=torch.cat([pooler(world[:,index]) for index,pooler in enumerate(self.poolers)]+[self.ego(ego)],dim=-1)
        state=self.fuse(state)
        return self.decoder(state+self.residual(state))

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def matched_single_layer_width():
    # Analytic parameter count avoids repeatedly instantiating large heads.
    def count(layers,width):
        pool=2*3072+(3072*width+width)+16*width+(4*width*width+4*width)+16+2*width
        ego=(20*128+128)+(128*128+128)
        fusion=(layers*width+128)*512+512
        residual=2*512+(512*1024+1024)+(1024*512+512)
        decoder=8*512+2*512+(512*256+256)+(256*3+3)
        return layers*pool+ego+fusion+residual+decoder
    target=count(3,512)
    width=min(range(512,1537,8),key=lambda value:abs(count(1,value)-target))
    assert abs(count(1,width)-target)/target<=.05
    return width

def build_head(kind):
    if kind=="ego":return MeanMLPReadout(ego_only=True)
    if kind=="linear":return LinearReadout()
    if kind=="fmlp":return MeanMLPReadout()
    if kind=="sf1":return StateFusionReadout(layers=1)
    if kind=="sf1-wide":return StateFusionReadout(layers=1,width=matched_single_layer_width())
    if kind=="sf3":return StateFusionReadout(layers=3)
    raise ValueError(kind)
