import math
from typing import Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint
from einops.layers.torch import Rearrange
from .unet import UNet
from .vit import ViT

class ARtransformer_shape(nn.Module):

    def __init__(self, params):
        super().__init__()
        # Read in the network specifications from the params
        self.params = params
        self.shape = self.params['shape'] # L,C,X,Y
        self.n_energy_layers = self.shape[0]

        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.shape[2] * self.shape[3] # X*Y
        self.dims_c = self.params["condition_dim"]
        self.bayesian = False

        self.c_embed = self.params.get("c_embed", None)
        self.x_embed = self.params.get("x_embed", None)

        self.encode_t_dim = self.params.get("encode_t_dim", 64)
        self.encode_t_scale = self.params.get("encode_t_scale", 30)

        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout_transformer", 0.0),
            # activation=params.get("activation", "relu"),
            batch_first=True,
        )
        if self.x_embed:
            inch = self.shape[1]
            ouch = params.get('x_embed_channels', 3)
            kernel = params.get('x_embed_kernel', (4,2))
            stride = params.get('x_embed_stride', (2,2))
            intermediate_dim = ouch * math.prod([
                1 + (l-k)//s for l, k, s in zip(self.shape[-2:], kernel, stride)
            ])
            self.x_embed = nn.Sequential(
                nn.Flatten(0, 1),
                nn.Conv2d(inch, ouch, kernel, stride),
                Rearrange('(b l) c x y -> b l (c x y)', l=self.n_energy_layers),
                nn.SiLU(),
                nn.Linear(intermediate_dim, self.dim_embedding),
            )
        if self.c_embed:
            self.c_embed = nn.Sequential(
                nn.Linear(1, self.dim_embedding),
                nn.SiLU(),
                nn.Linear(self.dim_embedding, self.dim_embedding)
            )
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.encode_t_dim, scale=self.encode_t_scale),
            nn.Linear(self.encode_t_dim, self.encode_t_dim)
        )
        self.subnet = self.build_subnet()
        self.positional_encoding = PositionalEncoding(
            d_model=self.dim_embedding, max_len=max(self.dims_in, self.dims_c) + 1, dropout=0.0
        )

    def compute_embedding(
        self, p: torch.Tensor, dim: int, embedding_net: Optional[nn.Module]
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(
            dim, device=p.device, dtype=p.dtype
        )[None, : p.shape[1], :].expand(p.shape[0], -1, -1)
        if embedding_net is None:
            n_rest = self.dim_embedding - dim - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=2)
        else:
            return self.positional_encoding(embedding_net(p))

    def build_subnet(self):
        subnet_config = self.params.get('subnet', 'UNet') 
        subnet_class = subnet_config['class']
        subnet_params = subnet_config['params']
        if subnet_class == 'UNet':
            return UNet(subnet_params)
        if subnet_class == 'ViT':
            return ViT(subnet_params)

    def sample_dimension(self, c: torch.Tensor):

        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        x_0 = torch.randn((batch_size, *self.shape[1:]), device=device, dtype=dtype)

        # NN wrapper to pass into ODE solver
        def net_wrapper(t, x_t):
            t_torch = t * torch.ones((batch_size, 1), dtype=dtype, device=device)
            v = self.subnet(x_t, t_torch, c.flatten(0,1))
            return v

        # Solve ODE from t=1 to t=0
        with torch.inference_mode():
            x_t = odeint(
                net_wrapper, x_0,torch.tensor([0, 1], dtype=dtype, device=device),
                **self.params.get("solver_kwargs", {})
            )
        # Extract generated samples and mask out masses if not needed
        x_1 = x_t[-1]

        return x_1.unsqueeze(1)

    def forward(self, c, x_t=None, t=None, x=None, rev=False):
        if not rev:

            if self.x_embed is None: x = x.flatten(2) # b l c x y -> b l (c x y)
            # xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
            xp = x
            embedding = self.transformer(
                src=self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed),
                tgt=self.compute_embedding(
                    xp, dim=self.n_energy_layers+1, embedding_net=self.x_embed,
                ),
                tgt_mask=torch.ones(
                    (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            
            x_t = x_t.flatten(0,1) # b l c x y -> (b l) c x y
            pred = self.subnet(x_t, t.reshape((-1, 1)), embedding.flatten(0,1))
            pred = pred.unflatten(0, (-1, self.n_energy_layers)) # (b l) c x y -> b l c x y
            
        else:
            x = torch.zeros((len(c), 1, *self.shape[1:]), device=c.device, dtype=c.dtype)
            c_embed = self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed)
            for i in range(self.n_energy_layers):
                if self.x_embed is None: x = x.flatten(2) # b l c x y -> b l (c x y)
                embedding = self.transformer(
                    src=c_embed,
                    tgt=self.compute_embedding(
                        x, dim=self.n_energy_layers+1, embedding_net=self.x_embed,
                    ),
                    tgt_mask=torch.ones(
                        (x.size(1), x.size(1)), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1),
                )
                x_new = self.sample_dimension(embedding[:, -1:,:])
                x = x.unflatten(2, self.shape[1:]) # b l (c x y) -> b l c x y
                x = torch.cat((x, x_new), dim=1)

            pred = x[:, 1:]

        return pred


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
