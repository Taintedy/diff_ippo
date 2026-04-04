import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        assert half_dim > 0, "Embedding dimension too small"
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):


    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        n_groups = min(n_groups, out_channels)

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('b c h -> b c 1 h'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('b c 1 h -> b c h'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)



def extract(a, t, x_shape):
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):

    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)



class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                pred.numpy().squeeze(),
                targ.numpy().squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}




def cumulative_visible_belief_sum(
    beliefs,        
    trajectories,   
    fov_deg=60.0,
    max_range=10.0
):


    device = beliefs.device
    dtype = beliefs.dtype

    if beliefs.ndim == 4:
        beliefs = beliefs[:, 0]   

    B, H, W = beliefs.shape
    _, T, _ = trajectories.shape

    fov_rad = math.radians(fov_deg) / 2.0

    x = trajectories[..., 0]                  
    y = trajectories[..., 1]                 
    theta = torch.atan2(
        trajectories[..., 2],
        trajectories[..., 3]
    )                                          

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )                                          

    xx = xx[None, None, :, :]                  
    yy = yy[None, None, :, :]                  

    x = x[:, :, None, None]                    
    y = y[:, :, None, None]                    
    theta = theta[:, :, None, None]            

    dx = xx - x                               
    dy = yy - y

    dist = torch.sqrt(dx**2 + dy**2)

    angles = torch.atan2(dy, dx)
    angle_diff = torch.atan2(
        torch.sin(angles - theta),
        torch.cos(angles - theta)
    )

    visible = (
        (dist <= max_range) &
        (angle_diff.abs() <= fov_rad)
    )                                          

    ever_visible = visible.any(dim=1)          

    belief_sum = (beliefs * ever_visible).sum(dim=(1, 2))

    return belief_sum



yy, xx = torch.meshgrid(
    torch.arange(256, device="cuda"),
    torch.arange(256, device="cuda"),
    indexing="ij"
)
xx = xx[None]
yy = yy[None]

def cumulative_visible_belief_sum_fast(
    beliefs,
    trajectories,
    fov_deg=60.0,
    max_range=10.0,
):
    if beliefs.ndim == 4:
        beliefs = beliefs[:, 0]

    device = beliefs.device
    B, H, W = beliefs.shape
    _, T, _ = trajectories.shape

    fov_rad = math.radians(fov_deg) / 2



    ever_visible = torch.zeros((B, H, W), device=device, dtype=torch.bool)

    x = trajectories[..., 0]
    y = trajectories[..., 1]
    theta = torch.atan2(trajectories[..., 2], trajectories[..., 3])

    for t in range(T): 
        xt = x[:, t][:, None, None]
        yt = y[:, t][:, None, None]
        thetat = theta[:, t][:, None, None]

        dx = xx - xt
        dy = yy - yt

        dist = torch.sqrt(dx**2 + dy**2)

        angles = torch.atan2(dy, dx)
        angle_diff = torch.atan2(
            torch.sin(angles - thetat),
            torch.cos(angles - thetat)
        )

        visible = (dist <= max_range) & (angle_diff.abs() <= fov_rad)
        ever_visible |= visible

    return (beliefs * ever_visible).sum(dim=(1, 2))



import torch
import math
def soft_visible_belief_loss_fast(
    beliefs,       
    trajectories,  
    fov_deg=60.0,
    max_range=10.0,
    alpha=10.0,
    beta=10.0,
):
    
    if beliefs.ndim == 4:
        beliefs = beliefs[:, 0]
    device = beliefs.device
    B, H, W = beliefs.shape
    _, T, _ = trajectories.shape

    visibility = torch.zeros((B, H, W), device=device)

    fov_rad = math.radians(fov_deg) / 2.0

    for t in range(T):
        x = trajectories[:, t, 0][:, None, None]
        y = trajectories[:, t, 1][:, None, None]
        theta = torch.atan2(
            trajectories[:, t, 2],
            trajectories[:, t, 3]
        )[:, None, None]

        dx = xx - x
        dy = yy - y

        dist = torch.sqrt(dx * dx + dy * dy)
        angle = torch.atan2(dy, dx)
        angle_diff = torch.atan2(
            torch.sin(angle - theta),
            torch.cos(angle - theta)
        )

        range_w = torch.sigmoid(beta * (max_range - dist))
        angle_w = torch.sigmoid(alpha * (fov_rad - angle_diff.abs()))
        v_t = range_w * angle_w  

        visibility = visibility.detach() + v_t - visibility.detach() * v_t

    loss = -(beliefs * visibility).sum(dim=(1, 2)).mean()
    return loss




def soft_visible_belief_loss(
    beliefs,
    trajectories,
    fov_deg=60.0,
    max_range=10.0,
    alpha=10.0,   
    beta=10.0    
):


    if beliefs.ndim == 4:
        beliefs = beliefs[:, 0]

    B, H, W = beliefs.shape
    _, T, _ = trajectories.shape
    device = beliefs.device

    x = trajectories[..., 0]
    y = trajectories[..., 1]
    theta = torch.atan2(trajectories[..., 2], trajectories[..., 3])

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    xx = xx[None, None]
    yy = yy[None, None]

    x = x[:, :, None, None]
    y = y[:, :, None, None]
    theta = theta[:, :, None, None]

    dx = xx - x
    dy = yy - y
    dist = torch.sqrt(dx**2 + dy**2)

    angles = torch.atan2(dy, dx)
    angle_diff = torch.atan2(
        torch.sin(angles - theta),
        torch.cos(angles - theta)
    )

    fov_rad = math.radians(fov_deg) / 2.0

    range_weight = torch.sigmoid(beta * (max_range - dist))
    angle_weight = torch.sigmoid(alpha * (fov_rad - angle_diff.abs()))

    visibility = range_weight * angle_weight   # (B,T,H,W)

    visibility = 1.0 - torch.prod(1.0 - visibility, dim=1)

    loss = -(beliefs * visibility).sum(dim=(1,2)).mean()

    return loss




def trajectory_smoothness_loss(traj):

    pos = traj[..., :2]

    vel = pos[:, 1:] - pos[:, :-1]

    acc = vel[:, 1:] - vel[:, :-1]

    jerk = acc[:, 1:] - acc[:, :-1]

    return jerk.pow(2).mean() + acc.pow(2).mean() + 5*vel.pow(2).mean()