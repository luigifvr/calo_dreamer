import torch
import torch.nn as nn
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
            self.c_encoding = nn.Linear(self.condition_dim, self.encode_t_dim)

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