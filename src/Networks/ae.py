from itertools import pairwise
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def add_coord_channels(x, break_dims=None):
    ndim = len(x.shape)
    channels = [x]
    for d in break_dims:
        coord = torch.linspace(0, 1, x.shape[d], device=x.device)
        cast_shape = np.where(np.arange(ndim) == d, -1, 1)
        expand_shape = np.where(np.arange(ndim) == 1, 1, x.shape)
        channels.append(coord.view(*cast_shape).expand(*expand_shape))
    return torch.cat(channels, dim=1)

# modified from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py
# allows for different implementation compared to unet
class Conv3DBlock(nn.Module):
    """
    Downsampling block for U-Net.

    __init__ parameters:
        in_channels  -- number of input channels
        out_channels -- desired number of output channels
        down_kernel  -- kernel for the downsampling operation
        down_stride  -- stride for the downsampling operation
        down_pad     -- size of the circular padding
        cond_dim     -- dimension of conditional input
        bottleneck   -- whether this is the bottlneck block
        break_dims   -- the index of dimensions at which translation symmetry
                        should be broken
    """

    def __init__(self, in_channels, out_channels, down_kernel=None, down_stride=None,
                 down_pad=None, cond_dim=None, bottleneck=False, break_dims=None):

        super(Conv3DBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        
        self.break_dims = break_dims or []
        self.conv1 = nn.Conv3d(
            in_channels=in_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.act = nn.SiLU()

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.Conv3d(
                in_channels=out_channels+len(self.break_dims), out_channels=out_channels,
                kernel_size=down_kernel, stride=down_stride, padding=down_pad
            )

    def forward(self, input, condition=None):

        # conv1
        res = add_coord_channels(input, self.break_dims)
        res = self.conv1(res)

        # conditioning
        if condition is not None:
            res = res + self.cond_layer(condition).view(
                -1, self.out_channels, 1, 1, 1
            )
        res = self.act(self.bn1(res))
        #res = self.bn1(self.act(res))

        # conv2
        res = add_coord_channels(res, self.break_dims)
        res = self.conv2(res)
        res = self.act(self.bn2(res))

        # pooling
        out = None
        if not self.bottleneck:
            out = add_coord_channels(res, self.break_dims)
            out = self.pooling(out)
        else:
            out = res
        return out, res

class UpConv3DBlock(nn.Module):

    """
    Upsampling block for U-Net.

    __init__ parameters:
        in_channels    -- number of input channels
        out_channels   -- desired number of output channels
        up_kernel      -- kernel for the upsampling operation
        up_stride      -- stride for the upsampling operation
        up_crop        -- size of cropping in the circular dimension
        cond_dim       -- dimension of conditional input
        output_padding -- argument forwarded to ConvTranspose
        break_dims   -- the index of dimensions at which translation symmetry
                should be broken
    """

    def __init__(self, in_channels, out_channels, up_kernel=None, up_stride=None,
                 up_crop=0, cond_dim=None, output_padding=0, break_dims=None):

        super(UpConv3DBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
 
        self.break_dims = break_dims or []
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=up_kernel, stride=up_stride, padding=up_crop,
            output_padding=output_padding
        )
        self.act = nn.SiLU()
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.conv1 = nn.Conv3d(
            in_channels=out_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=out_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=3, padding=1
        )

    def forward(self, input, residual=None, condition=None):

        # upsample
        out = add_coord_channels(input, self.break_dims)
        out = self.upconv1(out)

        # residual connection
        if residual != None:
            out = out + residual

        # conv1
        out = add_coord_channels(out, self.break_dims)
        out = self.conv1(out)

        # conditioning
        if condition is not None:
            out = out + self.cond_layer(condition).view(
                -1, self.out_channels, 1, 1, 1
            )
        out = self.act(self.bn1(out))

        # conv2
        out = add_coord_channels(out, self.break_dims)
        out = self.conv2(out)
        out = self.act(self.bn2(out))

        return out

class AutoEncoder(nn.Module):
    """
    :param param: A dictionary containing the relevant network parameters:
    """

    def __init__(self, param):

        super(AutoEncoder, self).__init__()

        defaults = {
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'ae_level_channels': [32, 1],
            'ae_level_kernels': [[3, 2, 3]],
            'ae_level_strides': [[3, 2, 3]],
            'ae_level_pads': [0],
            'ae_encode_c': False,
            'ae_encode_c_dim': 32,
            'ae_break_dims': None,
            'activation': nn.SiLU(),
            'ae_kl': False,
            'ae_latent_dim': 100,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)
        
        self.ae_break_dims = self.ae_break_dims or None

        # Conditioning
        self.total_condition_dim = self.ae_encode_c_dim if self.ae_encode_c else self.condition_dim

        if self.ae_encode_c_dim:
            self.c_encoding = nn.Sequential(
                nn.Linear(self.condition_dim, self.ae_encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.ae_encode_c_dim, self.ae_encode_c_dim)
            )

        *level_channels, bottle_channel = self.ae_level_channels

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            Conv3DBlock(
                n, m, self.ae_level_kernels[i], self.ae_level_strides[i],
                self.ae_level_pads[i], cond_dim=self.total_condition_dim,
                break_dims=self.ae_break_dims
            ) for i, (n, m) in enumerate(pairwise([self.in_channels] + level_channels))
        ])

        # Bottleneck block
        
        
        if self.ae_kl:
            self.conv_mu = nn.Conv3d(
                    in_channels=bottle_channel, out_channels=bottle_channel,
                    kernel_size=(1,1,1)
                    )
            self.conv_logvar = nn.Conv3d(
                    in_channels=bottle_channel, out_channels=bottle_channel,
                    kernel_size=(1,1,1),
                    )

            self.bottleneck = nn.ModuleList([
                nn.Conv3d(
                    in_channels=level_channels[-1]+len(self.ae_break_dims),
                    out_channels=bottle_channel, kernel_size=(1,1,1)
                ),
            ])
        else:
            self.bottleneck = nn.ModuleList([
                nn.Conv3d(
                    in_channels=level_channels[-1]+len(self.ae_break_dims),
                    out_channels=bottle_channel, kernel_size=(1,1,1)
                )
            ])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            UpConv3DBlock(
                n, m, self.ae_level_kernels[-1 -i], self.ae_level_strides[-1-i],
                self.ae_level_pads[-1-i], cond_dim=self.total_condition_dim,
                break_dims=self.ae_break_dims
            ) for i, (n, m) in enumerate(pairwise([bottle_channel] + level_channels[::-1]))
        ])

        # Output layer
        self.output_layer = nn.Conv3d(
            in_channels=level_channels[0]+len(self.ae_break_dims),
            out_channels=1, kernel_size=(1, 1, 1)
        )

    def forward(self, x, c=None):

        if self.ae_encode_c:
            c = self.c_encoding(c)

        z = self.encode(x, c=c)
        x = self.decode(z, c=c)

        return x


    def encode(self, x, c=None):

        out = x
        for down in self.down_blocks:
            out, _ = down(out, c)
        out = add_coord_channels(out, self.ae_break_dims)
        for btl in self.bottleneck:
            out = btl(out)
        if self.ae_kl:
            mu = self.conv_mu(out)
            logvar = self.conv_logvar(out)
            return mu, logvar
        return out

    def decode(self, z, c=None):

        out = z
        for up in self.up_blocks:
            out = up(out, residual=None, condition=c)
        out = add_coord_channels(out, self.ae_break_dims)
        out = self.output_layer(out)
        return torch.sigmoid(out)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z

class CylindricalAutoEncoder(nn.Module):
    """
    :param param: A dictionary containing the relevant network parameters:
    """

    def __init__(self, param):

        super(AutoEncoder, self).__init__()

        defaults = {
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'ae_level_channels': [32, 1],
            'ae_level_kernels': [[3, 2, 3]],
            'ae_level_strides': [[3, 2, 3]],
            'ae_level_pads': [0],
            'ae_encode_c': False,
            'ae_encode_c_dim': 32,
            'activation': nn.SiLU(),
            'break_dims': None,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        # Conditioning
        self.total_condition_dim = self.ae_encode_c_dim if self.ae_encode_c else self.condition_dim

        if self.ae_encode_c_dim:
            # self.c_encoding = nn.Linear(self.condition_dim, self.encode_c_dim)
            self.c_encoding = nn.Sequential(
                nn.Linear(self.condition_dim, self.ae_encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.ae_encode_c_dim, self.ae_encode_c_dim)
            )

        *level_channels, bottle_channel = self.ae_level_channels

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            CylindricalConv3DBlock(
                n, m, self.ae_level_kernels[i], self.ae_level_strides[i],
                self.ae_level_pads[i], cond_dim=self.total_condition_dim, break_dims=self.break_dims,
            ) for i, (n, m) in enumerate(pairwise([self.in_channels] + level_channels))
        ])

        # Bottleneck block
        self.bottleneck = nn.Conv3d(
                in_channels=level_channels[-1], out_channels=bottle_channel, kernel_size=(1,1,1)
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            CylindricalUpConv3DBlock(
                n, m, self.ae_level_kernels[-1 -i], self.ae_level_strides[-1-i],
                self.ae_level_pads[-1-i], cond_dim=self.total_condition_dim,
                break_dims=self.break_dims, use_circ_crop=True,
            ) for i, (n, m) in enumerate(pairwise([bottle_channel] + level_channels[::-1]))
        ])

        # Output layer
        self.output_layer = nn.Conv3d(
            in_channels=level_channels[0], out_channels=1, kernel_size=(1, 1, 1)
        )
    
    def forward(self, x, c=None):

        if self.ae_encode_c:
            c = self.c_encoding(c)

        out = x

        # down path
        for down in self.down_blocks:
            out, _ = down(out, c)

        # bottleneck
        out = self.bottleneck(
            add_coord_channels(out,break_dims=self.break_dims)
        )

        # up path
        for up in self.up_blocks:
            out = up(out, residual=None, condition=c)

        # output
        out = self.output_layer(
            add_coord_channels( out, break_dims=self.break_dims)
        )

        return out


