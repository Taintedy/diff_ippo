import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, hidden_dim * 2)
        )

    def forward(self, x, cond):
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return x * (1 + gamma[..., None]) + beta[..., None]


class BeliefEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, belief):
        return self.net(belief) 




class BeliefEncoderTokens(nn.Module):
    def __init__(self, dim, num_tokens=64):
        super().__init__()
        self.dim = dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1),
            nn.Mish(),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (int(num_tokens**0.5), int(num_tokens**0.5))
        )

    def forward(self, belief):

        x = self.conv(belief)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)  
        return x




class CrossAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden = dim_head * heads

        self.to_q = nn.Linear(dim, hidden, bias=False)
        self.to_kv = nn.Linear(dim, hidden * 2, bias=False)
        self.to_out = nn.Linear(hidden, dim)

        self.last_attn = None

    def forward(self, x, context):

        B, T, _ = x.shape
        H = self.heads

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = q.view(B, T, H, -1).transpose(1, 2)
        k = k.view(B, -1, H, q.shape[-1]).transpose(1, 2)
        v = v.view(B, -1, H, q.shape[-1]).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.last_attn = attn.detach()

        out = (attn @ v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.to_out(out)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()

        self.block1 = Conv1dBlock(in_ch, out_ch, 5)
        self.block2 = Conv1dBlock(out_ch, out_ch, 5)

        self.time_mlp = nn.Linear(embed_dim, out_ch)
        self.film = FiLM(embed_dim, out_ch)

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.block1(x)
        h = h + self.time_mlp(emb)[..., None]
        h = self.film(h, emb)
        h = self.block2(h)
        return h + self.residual(x)

class ConditionalTemporalUnet(nn.Module):
    def __init__(
        self,
        transition_dim=4,
        belief_dim=128,
        dim=64,
        dim_mults=(1, 2, 4, 8),

    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.belief_encoder = BeliefEncoder(belief_dim)
        self.context_proj = nn.Linear(belief_dim + dim, dim)

        dims = [transition_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = []

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, dim),
                ResidualTemporalBlock(dim_out, dim_out, dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]


        self.belief_token_encoder = BeliefEncoderTokens(mid_dim, num_tokens=1024)
        self.mid_cross_attn = CrossAttention(mid_dim)

        self.mid1 = ResidualTemporalBlock(mid_dim, mid_dim, dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) 
        self.mid2 = ResidualTemporalBlock(mid_dim, mid_dim, dim)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, dim),
                ResidualTemporalBlock(dim_in, dim_in, dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final = nn.Sequential(
            Conv1dBlock(dim, dim, 5),
            nn.Conv1d(dim, transition_dim, 1)
        )

    def forward(self, x, belief, time):


        x = einops.rearrange(x, 'b h t -> b t h')

        t_emb = self.time_mlp(time)
        b_emb = self.belief_encoder(belief)
        emb = self.context_proj(torch.cat([t_emb, b_emb], dim=-1))

        skips = []

        for block1, block2, attn, down in self.downs:
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            skips.append(x)
            x = down(x)


        belief_tokens = self.belief_token_encoder(belief)  

        x = self.mid1(x, emb)

        x = x.permute(0, 2, 1)  
        x = x + self.mid_cross_attn(x, belief_tokens)
        x = x.permute(0, 2, 1)

        # x = self.mid1(x, emb)
        x = self.mid_attn(x)
        x = self.mid2(x, emb)

        for block1, block2, attn, up in self.ups:
            skip = skips.pop()

            x = torch.cat([x, skip], dim=1)
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            x = up(x)


        x = self.final(x)

        return einops.rearrange(x, 'b t h -> b h t')
