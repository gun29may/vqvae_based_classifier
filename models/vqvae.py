
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder,Encoder1
from models.quantizer import VectorQuantizer
from models.decoder import Decoder,Decoder2
class VQVAE_new(nn.Module):
    """
    This is the full VQ-VAE model, which consists of an Encoder and a Decoder.
    The Encoder maps input x to latent space z, and the Decoder reconstructs 
    the original input x from the latent representation z, utilizing skip connections.
    
    Inputs:
    - in_dim : the input dimension for the encoder and decoder
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of residual layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(VQVAE_new, self).__init__()
        self.encoder = Encoder1(in_dim, h_dim, n_res_layers, res_h_dim)
        self.decoder = Decoder2(h_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, x):
        # Get the latent code and skip connections from the encoder
        latent_code, skips = self.encoder(x)

        # Pass the latent code and skip connections to the decoder
        reconstructed_x = self.decoder((latent_code, skips))

        return reconstructed_x


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = VQVAE(40, 128, 3, 64)
    encoder_out= encoder(x)
    # print(type(_[0]))
    print('Encoder out shape:', encoder_out.shape,)
