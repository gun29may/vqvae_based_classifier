
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack



class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        return self.conv_stack(x)

import torch
import torch.nn as nn

class Encoder1(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder1, self).__init__()
        kernel = 4
        stride = 2
        self.conv1 = nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.residual_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Save intermediate outputs for skip connections
        skip1 = self.relu(self.conv1(x))   # First convolution layer
        skip2 = self.relu(self.conv2(skip1))  # Second convolution layer
        skip3 = self.relu(self.conv3(skip2))  # Third convolution layer
        
        # Residual stack output
        out = self.residual_stack(skip3)

        # Return the final output and the skip connections
        return out, [skip1, skip2, skip3]


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder1(40, 128, 3, 64)
    encoder_out,_ = encoder(x)
    print(type(_[0]))
    print('Encoder out shape:', encoder_out.shape,)
