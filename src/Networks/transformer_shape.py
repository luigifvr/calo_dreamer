import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint
from .vblinear import VBLinear
import numpy as np
from einops import rearrange


class ARtransformer_shape(nn.Module):

    def __init__(self, params):
        super().__init__()
        # Read in the network specifications from the params
        self.params = params

        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.params["shape"][2] * self.params["shape"][3]
        self.dims_c = self.params["n_con"]
        self.n_energy_layers = self.params["shape"][0]
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
            self.x_embed = nn.Sequential(nn.Linear(1, self.dim_embedding),
                                     nn.Linear(self.dim_embedding, self.dim_embedding))
        if self.c_embed:
            self.c_embed = nn.Sequential(nn.Linear(1, self.dim_embedding),
                                               nn.Linear(self.dim_embedding, self.dim_embedding))
        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                     scale=self.encode_t_scale),
                                           nn.Linear(self.encode_t_dim, self.encode_t_dim))

        self.subnet = self.build_subnet()
        self.positional_encoding = PositionalEncoding(d_model=self.dim_embedding,
                                                      max_len=max(self.dims_in, self.dims_c) + 1,
                                                      dropout=0.0)
    def compute_embedding(
        self, p: torch.Tensor,
            dim: int,
            embedding_net: Optional[nn.Module]
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(dim, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if embedding_net is None:
            n_rest = self.dim_embedding - dim - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=2)
        else:
            return self.positional_encoding(embedding_net(p))

    def build_subnet(self):

        # self.intermediate_dim = self.params.get("intermediate_dim", 512)
        # self.dropout = self.params.get("dropout", 0.0)
        # self.activation = self.params.get("activation", "SiLU")
        # self.layers_per_block = self.params.get("layers_per_block", 8)
        # self.normalization = self.params.get("normalization", None)
        #
        # linear = nn.Linear(self.dim_embedding + self.encode_t_dim + self.dims_in, self.intermediate_dim)
        # layers = [linear, getattr(nn, self.activation)()]
        #
        # for _ in range(1, self.layers_per_block - 1):
        #     linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
        #     layers.append(linear)
        #     if self.normalization is not None:
        #         layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
        #     if self.dropout is not None:
        #         layers.append(nn.Dropout(p=self.dropout))
        #     layers.append(getattr(nn, self.activation)())
        #
        # linear = nn.Linear(self.intermediate_dim, 1)
        # layers.append(linear)
        #
        # return nn.Sequential(*layers)
        return SmallUNet(self.params, condition_dim=(self.encode_t_dim+self.dim_embedding))

    def sample_dimension(
            self, c: torch.Tensor):

        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        net = self.subnet
        x_0 = torch.randn((batch_size, 1, self.params["shape"][2],self.params["shape"][3]), device=device, dtype=dtype)
        #x_0 = torch.randn((batch_size, self.dims_in), device=device, dtype=dtype)

        # NN wrapper to pass into ODE solver
        def net_wrapper(t, x_t):
            #t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            t_torch = t * torch.ones((x_t.size(0), 1), dtype=dtype, device=device)
            t_torch = self.t_embed(t_torch)
            v = net(x_t, torch.cat([t_torch.reshape(batch_size, -1), c.squeeze()], dim=-1))
            # v = net(torch.cat([x_t, t_torch.reshape(batch_size, -1), c.squeeze()], dim=-1))
            return v

        # Solve ODE from t=1 to t=0
        with torch.no_grad():
            x_t = odeint(
                net_wrapper,
                x_0,
                torch.tensor([0, 1], dtype=dtype, device=device),
                **self.params.get("solver_kwargs", {})
            )
        # Extract generated samples and mask out masses if not needed
        x_1 = x_t[-1]

        return x_1.unsqueeze(1)

    def forward(self, c,x_t=None, t=None, x=None, rev=False):
        if not rev:
            x = rearrange(x, " b c l x y -> b (c l) (x y)")
            xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
            embedding = self.transformer(
                src=self.compute_embedding(
                    c,
                    dim=self.dims_c,
                    embedding_net=self.c_embed,
                ),
                tgt=self.compute_embedding(
                    xp,
                    dim=self.n_energy_layers + 1,
                    embedding_net=self.x_embed,
                ),
                tgt_mask=torch.ones(
                    (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            t = rearrange(t, "b l c x y -> b (l c) (x y)")
            t = self.t_embed(t)
            x_t = rearrange(x_t, "b l c x y -> (b l) c x y")
            pred = self.subnet(x_t, torch.cat([t, embedding], dim=-1))
            pred = rearrange(pred, "(b l) c x y -> b l c x y", l = 45)
        else:
            #x = torch.zeros((c.shape[0], 1, self.dims_in), device=c.device, dtype=c.dtype)
            x = torch.zeros((c.shape[0], 1, 1, self.params["shape"][2],self.params["shape"][3]), device=c.device, dtype=c.dtype)
            c_embed = self.compute_embedding(
            c, dim=self.dims_c, embedding_net=self.c_embed)
            for i in range(self.n_energy_layers):
                x = rearrange(x, " b c l x y -> b (c l) (x y)")
                embedding = self.transformer(
                    src=c_embed,
                    tgt=self.compute_embedding(
                        x,
                        dim=self.n_energy_layers  + 1,
                        embedding_net=self.x_embed,
                    ),
                    tgt_mask=torch.ones(
                        (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1),
                )
                x_new = self.sample_dimension(
                    embedding[:, -1:,:]
                )
                x = rearrange(x, " b (l c) (x y) -> b l c x y", c=1, x=self.params["shape"][2])
                #x_new = rearrange(x_new, "(b l) c x y -> b l c x y")
                x = torch.cat((x, x_new), dim=1)

            pred = x[:, 1:]
            # pred = rearrange(pred, "b l c x y -> b c l x y")
           # pred = rearrange(pred, "b (c l) (x y) -> b c l x y", c= 1)


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

class SmallUNet(nn.Module):

    def __init__(self, param, condition_dim=0):
        super(SmallUNet, self).__init__()

        defaults = {
            'in_channels': 1,
            'out_channels': 1,
            'level_channels': [128, 256],
            'cond_layers': 2,
            'activation': nn.SiLU(),
            'output_activation': None,
            'bayesian': False,
        }
        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        # Encoding/decoding blocks
        # level_1_chnls, level_2_chnls, bottleneck_chnl = self.level_channels
        level_1_chnls, bottleneck_chnl = self.level_channels
        self.a_block1 = Conv2DBlock(
            in_channels=self.in_channels, out_channels=level_1_chnls, cond_dim=condition_dim, cond_layers=self.cond_layers)
        # self.a_block2 = Conv2DBlock(
        #     in_channels=level_1_chnls, out_channels=level_2_chnls,
        #     cond_dim=condition_dim, cond_layers=self.cond_layers
        # )

        self.bottleNeck = Conv2DBlock(
            in_channels=level_1_chnls, out_channels=bottleneck_chnl,
            bottleneck=True, cond_dim=condition_dim, cond_layers=self.cond_layers
        )
        # self.bottleNeck_linear = nn.Sequential(nn.Linear(1024, 1024),nn.SiLU(),nn.Linear(1024,1024))
    # self.s_block2 = UpConv2DBlock(
        #     in_channels=bottleneck_chnl, out_channels=level_2_chnls,
        #     cond_dim=condition_dim, cond_layers=self.cond_layers)

        self.s_block1 = UpConv2DBlock(
            in_channels=bottleneck_chnl, out_channels=level_1_chnls,
            num_classes=self.out_channels, cond_dim=condition_dim, cond_layers=self.cond_layers,
            last_layer=True
        )
        self.kl = torch.zeros(())

    def forward(self, x, condition=None):
        # Analysis path forward feed
        out, residual_level1 = self.a_block1(x, condition)
        # out, residual_level2 = self.a_block2(out, condition)
        out, _ = self.bottleNeck(out, condition)

        # out = rearrange(out, "b c x y -> b (c x y)") + condition
        # out = self.bottleNeck_linear(out)
        # out = rearrange(out, "b (c x y) -> b c x y", x = 4, y=1)
        # Synthesis path forward feed
        # out = self.s_block2(out, residual_level2, condition)
        out = self.s_block1(out, residual_level1, condition)



        return out

class Conv2DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, cond_dim=None, cond_layers=1, bottleneck=False,
                 ):
        super(Conv2DBlock, self).__init__()
        self.out_channels = out_channels
        # self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.cond_block = self.make_condition_block(cond_dim=cond_dim, cond_layers=cond_layers)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool2d(kernel_size=(4, 9), stride=(4, 9))
            # self.pooling = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding=1,
            #                          stride=(2,3))

    def make_condition_block(self, cond_dim, cond_layers):
        # Use linear layers
        # linear = nn.Linear(cond_dim, cond_dim)
        # layers = [linear, nn.SiLU()]
        #
        # for _ in range(1, cond_layers - 1):
        #     linear = nn.Linear(cond_dim, cond_dim)
        #     layers.append(linear)
        #     layers.append(nn.SiLU())
        layers = []
        # linear = nn.Linear(cond_dim, self.out_channels)
        linear = nn.Linear(cond_dim, self.out_channels)
        layers.append(linear)


        return nn.Sequential(*layers)

    def forward(self, input, condition=None):

        res = self.conv1(input)
        if condition is not None:
            # res = res + self.cond_block(condition).view(-1, self.out_channels, 1, 1)
            res = res + self.cond_block(condition).view(-1, self.out_channels, 1, 1)
        # res = self.act(self.bn1(res))
        # res = self.act(self.bn2(self.conv2(res)))
        res = self.act(res)
        res = self.act(self.conv2(res))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv2DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, last_layer=False, cond_dim=None, cond_layers=1, num_classes=None,
                 ):
        super(UpConv2DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
                last_layer == True and num_classes != None), 'Invalid arguments'
        self.out_channels = out_channels
        # self.cond_layer = nn.Linear(cond_dim, out_channels)
        # self.cond_block = self.make_condition_block(cond_dim=cond_dim, cond_layers=cond_layers)
        self.cond_block = self.make_condition_block(cond_dim=cond_dim, cond_layers=cond_layers)
        self.upconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 9),
                                          stride=(4, 9))
        self.act = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=(1, 1))


    def make_condition_block(self, cond_dim, cond_layers):
        # Use linear layers
        # linear = nn.Linear(cond_dim, cond_dim)
        # layers = [linear, nn.SiLU()]
        #
        # for _ in range(1, cond_layers - 1):
        #     linear = nn.Linear(cond_dim, cond_dim)
        #     layers.append(linear)
        #     layers.append(nn.SiLU())
        layers = []
        linear = nn.Linear(cond_dim, self.out_channels)
        layers.append(linear)

        return nn.Sequential(*layers)

    def forward(self, input, residual=None, condition=None):

        # upsample
        out = self.upconv1(input)

        # residual connection
        if residual != None:
            out = out + residual
        out = self.conv1(out)

        if condition is not None:
            # out = out + self.cond_block(condition).view(-1, self.out_channels, 1, 1)
            out = out + self.cond_block(condition).view(-1, self.out_channels, 1, 1)
        # out = self.act(self.bn1(out))
        # out = self.act(self.bn2(self.conv2(out)))
        out = self.act(out)
        out = self.act(self.conv2(out))
        if self.last_layer:
            out = self.conv3(out)

        return out
