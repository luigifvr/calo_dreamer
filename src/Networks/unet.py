import torch
import torch.nn as nn
import torch.nn.functional as F
# from Networks.vblinear import VBLinear

class SimpleUNet(nn.Module):
    """A simple non-convolutional U-Net model."""

    def __init__(self, param):
        """
        :param param: A dictionary containing the relevant network parameters:
                      
                      dim -- The data dimension.
                      condition_dim -- Dimension of conditional input
                      hidden_dims -- Decreasing list of internal dimensions
                      activation -- Activation function for hidden layers
                      output_activation -- Activation function for output layer
                      bayesian -- Whether or not to use bayesian layers
                      encode_t -- Whether or not to embed the time input
                      encode_t_dim -- Dimension of the time embedding
                      encode_t -- Whether or not to embed the conditional input
                      encode_t_dim -- Dimension of the condition embedding            
        """
        
        super(SimpleUNet, self).__init__()

        defaults = {
            'dim': 368,
            'condition_dim': 0,
            'hidden_dims': [128, 64, 32],
            'activation': nn.SiLU(),
            'output_activation': None,
            'bayesian': False,
            'encode_t': False,
            'encode_t_dim': 64,
            'encode_c': False,
            'encode_c_dim': 64,
        }
        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)
        
        assert all((d < self.dim for d in self.hidden_dims)), \
        'Hidden dimensions must be smaller than the data dimension.'

        self.build_layers()
    
    def build_layers(self):
        # TODO: implement Bayesian layer

        # organise dimensions
        self.hidden_dims.sort(reverse=True)
        level_dims = [self.dim] + self.hidden_dims
        extra_dims = (self.encode_t_dim if self.encode_t else 1) \
                 + (self.encode_c_dim if self.encode_c else self.condition_dim)
        
        # construct layers
        encoding_layers, decoding_layers = [], []
        for i in range(len(self.hidden_dims)):
            dim_hi, dim_lo = level_dims[i:i+2]
            encoding_layers.append(nn.Linear(dim_hi + extra_dims, dim_lo))
            decoding_layers.insert(0, nn.Linear(dim_lo + extra_dims, dim_hi))
        self.encoding_layers = nn.ModuleList(encoding_layers)
        self.decoding_layers = nn.ModuleList(decoding_layers)

        # TODO: Add normalisation layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d+extra_dims)
            for d in level_dims[:-1] + level_dims[len(level_dims):0:-1]
        ])

        # construct condition encodings
        if self.encode_t:
            self.t_encoding = nn.Linear(1, self.encode_t_dim)
        if self.encode_c:
            self.c_encoding = nn.Linear(self.condition_dim, self.encode_c_dim)

    def forward(self, x, t, c=None):

        # handle conditional inputs
        if self.encode_t:
            t = self.t_encoding(t)
        if self.condition_dim == 0:
            condition = t
        else:
            if self.encode_c:
                c = self.c_encoding(c)
            condition = torch.cat([t, c], 1)
        
        self.kl = torch.zeros(())

        residuals = []
        # encode
        for i, layer in enumerate(self.encoding_layers):
            residuals.append(x)
            x = torch.cat([x, condition], 1)
            x = self.layer_norms[i](x)
            x = layer(x)
            
            x = self.activation(x)

        # decode
        for i, layer in enumerate(self.decoding_layers):
            x = torch.cat([x, condition], 1)
            x = self.layer_norms[i+len(self.hidden_dims)](x)
            x = layer(x)
            x += residuals.pop()
            if i+1 < len(self.hidden_dims):
                x = self.activation(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

# modified from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py
class Conv3DBlock(nn.Module):
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

    def __init__(self, in_channels, out_channels, cond_dim=None, bottleneck=False):
        super(Conv3DBlock, self).__init__()
        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.act = nn.SiLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(3, 2, 3), stride=(3, 2, 3))

    def forward(self, input, condition=None):

        res = self.conv1(input)
        if condition is not None:
            res = res + self.cond_layer(condition).view(-1, self.out_channels, 1, 1, 1)
        res = self.act(self.bn1(res))
        res = self.act(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
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

    def __init__(self, in_channels, out_channels, last_layer=False, cond_dim=None, num_classes=None):
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None), 'Invalid arguments'
        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 2, 3), stride=(3, 2, 3))
        self.act = nn.SiLU()
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.conv1 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=num_classes, kernel_size=(1, 1, 1))

    def forward(self, input, residual=None, condition=None):

        # upsample
        out = self.upconv1(input)

        # residual connection
        if residual != None:
            out = out + residual
        out = self.conv1(out)
        if condition is not None:
            out = out + self.cond_layer(condition).view(-1, self.out_channels, 1, 1, 1)
        out = self.act(self.bn1(out))
        out = self.act(self.bn2(self.conv2(out)))
        if self.last_layer:
            out = self.conv3(out)
        return out


class UNet(nn.Module):
    """
    :param param: A dictionary containing the relevant network parameters:
                  
                  dim -- The data dimension.
                  condition_dim -- Dimension of conditional input
                  in_channels -- Number of channels in the input
                  out_channels -- Number of channels in the network output
                  level_channels -- Number of channels at each level (count top-down)       
                  encode_t -- Whether or not to embed the time input
                  encode_t_dim -- Dimension of the time embedding
                  encode_t -- Whether or not to embed the conditional input
                  encode_t_dim -- Dimension of the condition embedding            
                  activation -- Activation function for hidden layers
                  output_activation -- Activation function for output layer
                  bayesian -- Whether or not to use bayesian layers
    """

    def __init__(self, param):

        super(UNet, self).__init__()

        defaults = {
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'level_channels': [16, 64, 128],
            'encode_t': False,
            'encode_t_dim': 32,
            'encode_t_scale': 30,
            'encode_c': False,
            'encode_c_dim': 32,
            'activation': nn.SiLU(),
            'output_activation': None,
            'bayesian': False
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
            self.c_encoding = nn.Linear(self.condition_dim, self.encode_c_dim)
            
        # Encoding/decoding blocks 
        level_1_chnls, level_2_chnls, bottleneck_chnl = self.level_channels
        self.a_block1 = Conv3DBlock(
            in_channels=self.in_channels, out_channels=level_1_chnls,
            cond_dim=self.total_condition_dim
        )
        self.a_block2 = Conv3DBlock(
            in_channels=level_1_chnls, out_channels=level_2_chnls,
            cond_dim=self.total_condition_dim
        )
        self.bottleNeck = Conv3DBlock(
            in_channels=level_2_chnls, out_channels=bottleneck_chnl,
            bottleneck=True, cond_dim=self.total_condition_dim
        )
        self.s_block2 = UpConv3DBlock(
            in_channels=bottleneck_chnl, out_channels=level_2_chnls,
            cond_dim=self.total_condition_dim
        )
        self.s_block1 = UpConv3DBlock(
            in_channels=level_2_chnls, out_channels=level_1_chnls,
            num_classes=self.out_channels, cond_dim=self.total_condition_dim,
            last_layer=True
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

        #Analysis path forward feed
        out, residual_level1 = self.a_block1(x, condition)
        out, residual_level2 = self.a_block2(out, condition)
        out, _ = self.bottleNeck(out, condition)
        
        # Synthesis path forward feed
        out = self.s_block2(out, residual_level2, condition)
        out = self.s_block1(out, residual_level1, condition)
        
        return out


class PaddedConv3DBlock(nn.Module):
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

    def __init__(self, in_channels, out_channels, cond_dim=None, bottleneck=False) -> None:

        super(PaddedConv3DBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.act = nn.SiLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=3, stride=(3, 2, 2))

    def forward(self, input, condition=None):

        res = self.conv1(input)
        if condition is not None:
            res = res + \
                self.cond_layer(condition).view(-1, self.out_channels, 1, 1, 1)
        res = self.act(self.bn1(res))
        res = self.act(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class PaddedUpConv3DBlock(nn.Module):
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

    def __init__(self, in_channels, out_channels, last_layer=False, cond_dim=None, num_classes=None):

        super(PaddedUpConv3DBlock, self).__init__()

        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None), 'Invalid arguments'

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            stride=(3, 2, 2)
        )
        self.act = nn.SiLU()
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.conv1 = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            padding=1
        )
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(
                in_channels=out_channels, out_channels=num_classes,
                kernel_size=(1, 1, 1)
            )

    def forward(self, input, residual=None, condition=None):

        # upsample
        out = self.upconv1(input)

        # residual connection
        if residual != None:
            out = out + residual
        out = self.conv1(out)
        if condition is not None:
            out = out + \
                self.cond_layer(condition).view(-1, self.out_channels, 1, 1, 1)
        out = self.act(self.bn1(out))
        out = self.act(self.bn2(self.conv2(out)))
        if self.last_layer:
            out = self.conv3(out)
        return out
    

class PaddedUNet(nn.Module):
    """
    :param param: A dictionary containing the relevant network parameters:
                  
                  dim -- The data dimension.
                  condition_dim -- Dimension of conditional input
                  in_channels -- Number of channels in the input
                  out_channels -- Number of channels in the network output
                  level_channels -- Number of channels at each level (count top-down)       
                  encode_t -- Whether or not to embed the time input
                  encode_t_dim -- Dimension of the time embedding
                  encode_t -- Whether or not to embed the conditional input
                  encode_t_dim -- Dimension of the condition embedding            
                  activation -- Activation function for hidden layers
                  output_activation -- Activation function for output layer
                  bayesian -- Whether or not to use bayesian layers
    """

    def __init__(self, param):

        super(PaddedUNet, self).__init__()

        defaults = {
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'level_channels': [16, 64, 128],
            'encode_t': False,
            'encode_t_dim': 32,
            'encode_t_scale': 30,
            'encode_c': False,
            'encode_c_dim': 32,
            'activation': nn.SiLU(),
            'output_activation': None,
            'bayesian': False
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
            self.c_encoding = nn.Linear(self.condition_dim, self.encode_c_dim)

        # Encoding/decoding blocks
        level_1_chnls, level_2_chnls, bottleneck_chnl = self.level_channels
        self.a_block1 = PaddedConv3DBlock(
            in_channels=self.in_channels, out_channels=level_1_chnls,
            cond_dim=self.total_condition_dim,
        )
        self.a_block2 = PaddedConv3DBlock(
            in_channels=level_1_chnls, out_channels=level_2_chnls,
            cond_dim=self.total_condition_dim,
        )
        self.bottleNeck = PaddedConv3DBlock(
            in_channels=level_2_chnls, out_channels=bottleneck_chnl,
            bottleneck=True, cond_dim=self.total_condition_dim
        )
        self.s_block2 = PaddedUpConv3DBlock(
            in_channels=bottleneck_chnl, out_channels=level_2_chnls,
            cond_dim=self.total_condition_dim,
        )
        self.s_block1 = PaddedUpConv3DBlock(
            in_channels=level_2_chnls, out_channels=level_1_chnls,
            num_classes=self.out_channels, last_layer=True,
            cond_dim=self.total_condition_dim,
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

        x = F.pad(x, (0, 0, 1, 2, 0, 0))
        # downsample 1
        out, residual_level1 = self.a_block1(x, condition)
        out = F.pad(out, (0, 1, 0, 0, 0, 0))
        # downsample 2
        out, residual_level2 = self.a_block2(out, condition)
        # bottleneck
        out, _ = self.bottleNeck(out, condition)
        # upsample 1
        out = self.s_block2(out, residual_level2, condition)
        out = out[..., :-1]
        # upsample 2
        out = self.s_block1(out, residual_level1, condition)
        out = out[..., 1:-2, :]

        return out
    

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2)
                              * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)