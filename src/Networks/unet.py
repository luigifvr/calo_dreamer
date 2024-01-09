from itertools import pairwise
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from Networks.vblinear import VBLinear

class GaussianFourierProjection(nn.Module): # TODO: Move this (and defn in resnet.py) to separate file
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)


# modified from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py
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
    """

    def __init__(self, in_channels, out_channels, down_kernel=None, down_stride=None,
                 down_pad=None, cond_dim=None, bottleneck=False):

        super(Conv3DBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.act = nn.SiLU()

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(
                kernel_size=down_kernel, stride=down_stride, padding=down_pad
            )

    def forward(self, input, condition=None):

        # conv1
        res = self.conv1(input)

        # conditioning
        if condition is not None:
            res = res + self.cond_layer(condition).view(
                -1, self.out_channels, 1, 1, 1
            )
        res = self.act(self.bn1(res))

        # conv2
        res = self.conv2(res)
        res = self.act(self.bn2(res))

        # pooling
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
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
        
    """

    def __init__(self, in_channels, out_channels, up_kernel=None, up_stride=None,
                 up_crop=0, cond_dim=None, output_padding=0):

        super(UpConv3DBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=up_kernel, stride=up_stride, padding=up_crop,
            output_padding=output_padding
        )
        self.act = nn.SiLU()
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.conv1 = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1
        )

    def forward(self, input, residual=None, condition=None):

        # upsample
        out = self.upconv1(input)

        # residual connection
        if residual != None:
            out = out + residual

        # conv1
        out = self.conv1(out)

        # conditioning
        if condition is not None:
            out = out + self.cond_layer(condition).view(
                -1, self.out_channels, 1, 1, 1
            )
        out = self.act(self.bn1(out))

        # conv2
        out = self.conv2(out)
        out = self.act(self.bn2(out))

        return out


class UNet(nn.Module):
    """
    :param param: A dictionary containing the relevant network parameters:
                  
                  dim -- The data dimension.
                  condition_dim  -- Dimension of conditional input
                  in_channels    -- Number of channels in the input
                  out_channels   -- Number of channels in the network output
                  level_channels -- Number of channels at each level (count top-down)
                  level_kernels  -- Kernel shape for the up/down sampling operations
                  level_strides  -- Stride shape for the up/down sampling operations
                  level_pads     -- Padding for the up/down sampling operations
                  encode_t       -- Whether or not to embed the time input
                  encode_t_dim   -- Dimension of the time embedding
                  encode_t_scale -- Scale for the Gaussian Fourier projection
                  encode_c       -- Whether or not to embed the conditional input
                  encode_c_dim   -- Dimension of the condition embedding            
                  activation     -- Activation function for hidden layers
                  bayesian       -- Whether or not to use bayesian layers
    """

    def __init__(self, param):

        super(UNet, self).__init__()

        defaults = {
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'level_channels': [32, 64, 128],
            'level_kernels': [[3, 2, 3], [3, 2, 3]],
            'level_strides': [[3, 2, 3], [3, 2, 3]],
            'level_pads': [0, 0],
            'encode_t': False,
            'encode_t_dim': 32,
            'encode_t_scale': 30,
            'encode_c': False,
            'encode_c_dim': 32,
            'activation': nn.SiLU(),
            'bayesian': False,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        # Conditioning
        self.total_condition_dim = (self.encode_t_dim if self.encode_t else 1) \
            + (self.encode_c_dim if self.encode_c else self.condition_dim)

        if self.encode_t_dim:
            fourier_proj = GaussianFourierProjection(
                embed_dim=self.encode_t_dim, scale=self.encode_t_scale
            )
            self.t_encoding = nn.Sequential(
                fourier_proj, nn.Linear(self.encode_t_dim, self.encode_t_dim)
            )
        if self.encode_c_dim:
            # self.c_encoding = nn.Linear(self.condition_dim, self.encode_c_dim)
            self.c_encoding = nn.Sequential(
                nn.Linear(self.condition_dim, self.encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.encode_c_dim, self.encode_c_dim)
            )

        *level_channels, bottle_channel = self.level_channels

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            Conv3DBlock(
                n, m, self.level_kernels[i], self.level_strides[i],
                self.level_pads[i], cond_dim=self.total_condition_dim
            ) for i, (n, m) in enumerate(pairwise([self.in_channels] + level_channels))
        ])

        # Bottleneck block
        self.bottleneck_block = Conv3DBlock(
            level_channels[-1], bottle_channel, bottleneck=True,
            cond_dim=self.total_condition_dim,
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            UpConv3DBlock(
                n, m, self.level_kernels[-1 -i], self.level_strides[-1-i],
                self.level_pads[-1-i], cond_dim=self.total_condition_dim
            ) for i, (n, m) in enumerate(pairwise([bottle_channel] + level_channels[::-1]))
        ])

        # Output layer
        self.output_layer = nn.Conv3d(
            in_channels=level_channels[0], out_channels=1, kernel_size=(1, 1, 1)
        )

        self.kl = torch.zeros(())

    def forward(self, x, t, c=None):

        if self.encode_t:
            t = self.t_encoding(t)
        if c is None:
            condition = t
        else:
            if self.encode_c:
                c = self.c_encoding(c)
            condition = torch.cat([t, c], 1)

        residuals = []
        out = x

        # down path
        for down in self.down_blocks:
            out, res = down(out, condition)
            residuals.append(res)

        # bottleneck
        out, _ = self.bottleneck_block(out, condition)

        # up path
        for up in self.up_blocks:
            out = up(out, residuals.pop(), condition)

        # output
        out = self.output_layer(out)

        return out
    

class CylindricalConv3DBlock(nn.Module):
    """
    Downsampling block for cylindrical U-Net. These use circular padding in the
    angular direction prior to convolution. Convolutions add channels corresponding
    to dimensions in which the operations should not be translation equivariant.

    __init__ parameters:
        in_channels -- number of input channels
        out_channels -- desired number of output channels
        down_kernel -- kernel for the downsampling operation
        down_stride -- stride for the downsampling operation
        down_pad -- size of the circular padding
        cond_dim -- dimension of conditional input
        break_dims -- the index of dimensions at which translation symmetry should be broken
        bottleneck -- whether this is the bottlneck block
    """

    def __init__(self, in_channels, out_channels, down_kernel=None, down_stride=None,
                 down_pad=None, cond_dim=None, break_dims=None, bottleneck=False):
        
        super(CylindricalConv3DBlock, self).__init__()
        
        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.break_dims = break_dims or []
        self.conv1 = nn.Conv3d(
            in_channels=in_channels+len(self.break_dims),
            out_channels=out_channels, kernel_size=3, padding=(1, 0, 1)
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels+len(self.break_dims),
            out_channels=out_channels, kernel_size=3, padding=(1, 0, 1)
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.act = nn.SiLU()
        self.down_pad = down_pad
        self.circ_pad = lambda x, p: F.pad(x, (0, 0, p, p, 0, 0), mode='circular')

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=down_kernel, stride=down_stride)

    def forward(self, input, condition=None):

        # conv1
        res = self.circ_pad(input, 1)
        res = add_coord_channels(res, self.break_dims)
        res = self.conv1(res)

        # conditioning
        if condition is not None:
            res = res + \
                self.cond_layer(condition).view(-1, self.out_channels, 1, 1, 1)
        res = self.act(self.bn1(res))

        # conv2
        res = self.circ_pad(res, 1)
        res = add_coord_channels(res, self.break_dims)
        res = self.conv2(res)
        res = self.act(self.bn2(res))

        out = None
        if not self.bottleneck:
            out = self.pooling(self.circ_pad(res, self.down_pad))

        else:
            out = res
        return out, res


class CylindricalUpConv3DBlock(nn.Module):

    """
    Upsampling block for cylindrical U-Net. These use circular padding in the
    angular direction prior to convolutions. No padding is used before
    ConvTranpose operations, but the output is cropped. Convolutions add channels
    corresponding to dimensions in which the operations should not be translation equivariant.

    Downsampling block for cylindrical U-Net. These use circular padding in the
    angular direction prior to convolution. 

    __init__ parameters:
        in_channels -- number of input channels
        out_channels -- desired number of output channels
        up_kernel -- kernel for the upsampling operation
        up_stride -- stride for the upsampling operation
        up_crop -- size of cropping in the circular dimension
        use_circ_crop -- whether to average opposite ends when cropping
        last_layer -- whether this is the last block
        cond_dim -- dimension of conditional input
        num_classes -- specifies the number of output channels for dispirate
                       classes
        output_padding -- argument forwarded to ConvTranspose
        break_dims -- the index of dimensions at which translation symmetry should be broken
        
    """

    def __init__(self, in_channels, out_channels, up_kernel=None, up_stride=None,
                 last_layer=False, cond_dim=None, num_classes=None,
                 output_padding=0, up_crop=0, use_circ_crop=False, break_dims=None):
        
        super(CylindricalUpConv3DBlock, self).__init__()
        
        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None), 'Invalid arguments'
        
        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.break_dims = break_dims or []
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=up_kernel, stride=up_stride, output_padding=output_padding
        )
        self.act = nn.SiLU()
        self.up_crop = up_crop
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.conv1 = nn.Conv3d(
            in_channels=out_channels+len(self.break_dims),
            out_channels=out_channels, kernel_size=3, padding=(1, 0, 1)
        )
        self.conv2 = nn.Conv3d(
            in_channels=out_channels+len(self.break_dims),
            out_channels=out_channels, kernel_size=3, padding=(1, 0, 1)
        )
        self.circ_pad = lambda x, p: F.pad(
            x, (0, 0, p, p, 0, 0), mode='circular')
        self.use_circ_crop = use_circ_crop

        self.last_layer = last_layer

        if last_layer:
            self.conv3 = nn.Conv3d(
                in_channels=out_channels+len(self.break_dims),
                out_channels=num_classes, kernel_size=(1, 1, 1)
            )

    def circ_crop(self, x):
        """
        Cropping operation that averages over cirular padding
                L0 | L1 | X0 | X1 | ... | X7 | X8 | R0 | R1
                                     |
                                     V
            (X0+R0)/2 | (X1+R1)/2 | ... | (X7+L0)/2 | (X8+L1)/2
        """
        C = self.up_crop

        # store edges
        l_edge = x[..., :C, :]
        r_edge = x[..., -C:, :]

        # crop
        x = x[..., C:-C, :]

        # average with cropped edges
        x[..., :C, :] = (x[..., :C, :] + r_edge)/2
        x[..., -C:, :] = (x[..., -C:, :] + l_edge)/2

        return x

    def forward(self, input, residual=None, condition=None):

        # upsample
        input = add_coord_channels(input, self.break_dims)
        out = self.upconv1(input)
        if self.up_crop != 0:
            if self.use_circ_crop:
                out = self.circ_crop(out)
            else:
                out = out[..., self.up_crop:-self.up_crop, :]

        # residual connection
        if residual != None:
            out = out + residual

        # conv1
        out = self.circ_pad(out, 1)
        out = add_coord_channels(out, self.break_dims)
        out = self.conv1(out)

        # conditioning
        if condition is not None:
            out = out + self.cond_layer(condition).view(-1, self.out_channels, 1, 1, 1)
        out = self.act(self.bn1(out))

        # conv2
        out = self.circ_pad(out, 1)
        out = add_coord_channels(out, self.break_dims)
        out = self.conv2(out)
        out = self.act(self.bn2(out))

        if self.last_layer:
            out = add_coord_channels(out, self.break_dims)
            out = self.conv3(out)

        return out


class CylindricalUNet(nn.Module):
    """
    :param param: A dictionary containing the relevant network parameters:
                  
                  dim -- The data dimension.
                  condition_dim -- Dimension of conditional input
                  in_channels -- Number of channels in the input
                  out_channels -- Number of channels in the network output
                  level_channels -- Number of channels at each level (count top-down)
                  break_dims -- Dimensions at which translational symmetry should be broken
                  encode_t -- Whether or not to embed the time input
                  encode_t_dim -- Dimension of the time embedding
                  encode_t -- Whether or not to embed the conditional input
                  encode_t_dim -- Dimension of the condition embedding            
                  activation -- Activation function for hidden layers
                  output_activation -- Activation function for output layer
                  bayesian -- Whether or not to use bayesian layers
    """

    def __init__(self, param):

        super(CylindricalUNet, self).__init__()

        defaults = {
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'level_channels': [32, 64, 128],
            'break_dims': None,
            'encode_t': False,
            'encode_t_dim': 32,
            'encode_t_scale': 30,
            'encode_c': False,
            'encode_c_dim': 32,
            'activation': nn.SiLU(),
            'output_activation': None,
            'bayesian': False,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        # Conditioning
        self.total_condition_dim = (self.encode_t_dim if self.encode_t else 1) \
            + (self.encode_c_dim if self.encode_c else self.condition_dim)

        if self.encode_t_dim:
            fourier_proj = GaussianFourierProjection(
                embed_dim=self.encode_t_dim, scale=self.encode_t_scale
            )
            self.t_encoding = nn.Sequential(
                fourier_proj, nn.Linear(self.encode_t_dim, self.encode_t_dim)
            )
        if self.encode_c_dim:
            # self.c_encoding = nn.Linear(self.condition_dim, self.encode_c_dim)
            self.c_encoding = nn.Sequential(
                nn.Linear(self.condition_dim, self.encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.encode_c_dim, self.encode_c_dim)
            )

        # Encoding/decoding blocks
        level_1_chnls, level_2_chnls, bottleneck_chnl = self.level_channels
        self.a_block1 = CylindricalConv3DBlock(
            in_channels=self.in_channels, out_channels=level_1_chnls,
            down_kernel=(3, 4, 3), down_stride=(3, 2, 3), down_pad=2,
            cond_dim=self.total_condition_dim, break_dims=self.break_dims,
        )
        self.a_block2 = CylindricalConv3DBlock(
            in_channels=level_1_chnls, out_channels=level_2_chnls,
            down_kernel=(3, 3, 3), down_stride=(3, 2, 3), down_pad=1,
            cond_dim=self.total_condition_dim, break_dims=self.break_dims,
        )
        self.bottleNeck = CylindricalConv3DBlock(
            in_channels=level_2_chnls, out_channels=bottleneck_chnl,
            bottleneck=True, cond_dim=self.total_condition_dim,
            break_dims=self.break_dims,
        )
        self.s_block2 = CylindricalUpConv3DBlock(
            in_channels=bottleneck_chnl, out_channels=level_2_chnls,
            up_kernel=(3, 3, 3), up_stride=(3, 2, 3), up_crop=1,
            cond_dim=self.total_condition_dim,
            use_circ_crop=True, break_dims=self.break_dims,
        )
        self.s_block1 = CylindricalUpConv3DBlock(
            in_channels=level_2_chnls, out_channels=level_1_chnls,
            up_kernel=(3, 4, 3), up_stride=(3, 2, 3), up_crop=2,
            cond_dim=self.total_condition_dim,
            break_dims=self.break_dims, use_circ_crop=True,
            num_classes=self.out_channels, last_layer=True
        )
        self.kl = torch.zeros(())

    def forward(self, x, t, c=None):

        if self.encode_t:
            t = self.t_encoding(t)
        if c is None:
            condition = t
        else:
            if self.encode_c:
                c = self.c_encoding(c)
            condition = torch.cat([t, c], 1)

        # downsample 1
        out, residual_level1 = self.a_block1(x, condition)
        # downsample 2
        out, residual_level2 = self.a_block2(out, condition)

        # bottleneck
        out, _ = self.bottleNeck(out, condition)

        # upsample 1
        out = self.s_block2(out, residual_level2, condition)
        # upsample 2
        out = self.s_block1(out, residual_level1, condition)

        return out


def add_coord_channels(x, break_dims=None):
    ndim = len(x.shape)  # TODO: move to init? and other optimisations
    channels = [x]
    for d in break_dims:
        coord = torch.linspace(0, 1, x.shape[d], device=x.device)
        cast_shape = np.where(np.arange(ndim) == d, -1, 1)
        expand_shape = np.where(np.arange(ndim) == 1, 1, x.shape)
        channels.append(coord.view(*cast_shape).expand(*expand_shape))
    return torch.cat(channels, dim=1)
