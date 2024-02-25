"""Modified from github.com/facebookresearch/DiT/blob/main/models.py""" 

import math
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  

class ViT(nn.Module):
    """
    Vision transformer-based diffusion network.
    """
    # TODO: Embed time and condition separately into hidden_dim//2 then concat
    # TODO: Implement padding

    def __init__(self, param):

        super(ViT, self).__init__()

        defaults = {
            'shape': (1, 45, 16, 9),
            'patch_shape': (3, 2, 1),
            'condition_dim': 46,
            'hidden_dim': 24 * 24,
            'depth': 4,
            'num_heads': 4,
            'mlp_ratio': 4.0,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)
        
        C, L, A, R = self.shape
        assert L % self.patch_shape[0] == 0, \
            f"Input layer count ({L}) should be divisible by patch size ({self.patch_shape[0]})."
        assert A % self.patch_shape[1] == 0, \
            f"Input angular bin count ({A}) should be divisible by patch size ({self.patch_shape[1]})."
        assert R % self.patch_shape[2] == 0, \
            f"Input radial bin count ({R}) should be divisible by patch size ({self.patch_shape[2]})."
        assert not self.hidden_dim % 6, "Hidden dim should be divisible by 6 (for fourier position embeddings)"
        
        self.in_channels = C
        self.out_channels = C
        self.patch_shape = list(self.patch_shape)
        self.num_patches = [
            s // p for s, p in zip(self.shape[1:], self.patch_shape)
        ]

        # initialize x,t,c embeddings
        patch_dim = math.prod(self.patch_shape) * self.out_channels
        self.x_embedder = nn.Sequential(
            Rearrange(
                'b c (l p1) (a p2) (r p3) -> b (l a r) (p1 p2 p3 c)',
                **dict(zip(('p1', 'p2', 'p3'), self.patch_shape))
            ),
            nn.Linear(patch_dim, self.hidden_dim),
        )
        self.t_embedder = TimestepEmbedder(self.hidden_dim)
        self.c_embedder = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # compute fixed position embedding
        self.pos_embed = nn.Parameter(
            self.get_cylindrical_sincos_pos_embed(self.num_patches, self.hidden_dim),
            requires_grad=False
        )

        # initialize transformer stack
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_dim, self.num_heads, mlp_ratio=self.mlp_ratio
            ) for _ in range(self.depth)
        ])

        # initialize output layer
        self.final_layer = FinalLayer(
            self.hidden_dim, self.patch_shape, self.out_channels
        )

        # custom weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Forward pass of DiT.
        x: (B, C, L, A, R) tensor of spatial inputs
        t: (B,) tensor of diffusion timesteps
        c: (B, K) tensor of conditions
        """
        x = self.x_embedder(x) + self.pos_embed  # (B, T, D), where T = (L*A*R)/prod(patch_size)
        t = self.t_embedder(t)                   # (B, D)
        c = self.c_embedder(c)                   # (B, D)
        c = t + c                                # (B, D)
        for block in self.blocks:
            x = block(x, c)                      # (B, T, D)
        x = self.final_layer(x, c)               # (B, T, prod(patch_shape) * out_channels)
        x = rearrange(                           # (B, out_channels, L, A, R)
            x, 'b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)',
            **dict(zip(('l', 'a', 'r', 'p1', 'p2', 'p3'), self.num_patches+self.patch_shape))
        )                   
        return x

    @staticmethod
    def get_cylindrical_sincos_pos_embed(num_patches, dim, temperature=10000):    

        L, A, R = num_patches
        z, alpha, r = torch.meshgrid(
            torch.arange(L)/L,
            torch.arange(A)*(2*math.pi/A),
            torch.arange(R)/R,
            indexing='ij'
        )
        x = r*alpha.cos()
        y = r*alpha.sin()

        fourier_dim = dim // 6
        omega = torch.arange(fourier_dim) / (fourier_dim - 1)
        omega = 1. / (temperature ** omega)    
        z = z.flatten()[:, None] * omega[None, :]
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        
        pe = torch.cat(
            (x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1
        )
        # pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
        return pe        


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, patch_shape, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, math.prod(patch_shape) * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb        