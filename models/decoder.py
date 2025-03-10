
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack
from torchsummary import summary


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
import torch
import torch.nn as nn

class Decoder1(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z, p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder1, self).__init__()
        kernel = 4
        stride = 2

        self.conv_transpose1 = nn.ConvTranspose2d(
            in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.residual_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        
        self.conv_transpose2 = nn.ConvTranspose2d(
            h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(
            h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, skips):
        """
        Forward pass through the decoder, incorporating skip connections.

        Inputs:
        - x : the input latent code
        - skips : a list of skip connections from the encoder (from high to low resolution)

        """

        # Decoder starts with the latent code
        x = self.relu(self.conv_transpose1(x))
        x = x + skips[2]

        x = self.residual_stack(x)

        x = self.relu(self.conv_transpose2(x))
        x = x + skips[1]

        x = self.conv_transpose3(x)
        x = x + skips[0]

        return x

class Decoder2(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder2, self).__init__()
        kernel = 4
        stride = 2

        # Conv layers for processing the decoder steps
        self.conv_transpose1 = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.residual_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        self.conv_transpose2 = nn.ConvTranspose2d(h_dim * 2, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d((h_dim // 2) * 2, 3, kernel_size=kernel, stride=stride, padding=1)

        # Conv layers to match the dimensions of skip connections
        self.skip1_conv = nn.Conv2d(128, 64, kernel_size=1)  # Adjust skip[1] from 128 to 64
        self.skip0_conv = nn.Conv2d(64, 3, kernel_size=1)    # Adjust skip[0] to match output channels

        # Conv to reduce the channels before the residual block after concatenation
        self.conv_reduce = nn.Conv2d(h_dim * 2, h_dim, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        Accept a tuple (x, skips) instead of multiple separate arguments.
        """
        x, skips = inputs  # Unpack the inputs

        # Decoder starts with the latent code
        x = self.relu(self.conv_transpose1(x))

        # Concatenate the highest resolution skip connection
        x = torch.cat([x, skips[2]], dim=1)

        # Reduce the channels before passing to the residual block
        x = self.conv_reduce(x)

        # Pass through the residual block
        x = self.residual_stack(x)

        # Concatenate the next skip connection, but first adjust its size
        x = self.relu(self.conv_transpose2(x))
        skips[1] = self.skip1_conv(skips[1])  # Adjust skip[1] channels from 128 to 64
        x = torch.cat([x, skips[1]], dim=1)

        # Final decoding stage with the lowest resolution skip connection
        skips[0] = self.skip0_conv(skips[0])  # Adjust skip[0] channels to match output (3 channels)
        x = self.conv_transpose3(x)
        x = torch.cat([x, skips[0]], dim=1)

        return x



if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder2(40, 128, 3, 64)
    # summary(decoder, [(40, 16, 16), [(64, 64, 64), (128, 32, 32), (128, 16, 16)]])
    x_input = torch.randn(1, 40, 16, 16)  # Latent code input
    skip_connections = [
    torch.randn(1, 64, 64, 64),  # Skip connection from an earlier stage of encoder
    torch.randn(1, 128, 32, 32),  # Another skip connection
    torch.randn(1, 128, 16, 16)]   # Last skip connection]

# Perform the forward pass
    output = decoder((x_input, skip_connections))

    # decoder_out = decoder(x,[1,1,1])
    # print('Dncoder out shape:', decoder_out.shape)
