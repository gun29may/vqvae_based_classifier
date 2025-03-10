import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from models.vqvae import VQVAE
import os
parser = argparse.ArgumentParser()
from tqdm import tqdm
import torch.nn.functional as F
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image
    # Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])
# Path to the directory containing images
img_dir = '/home/gunmay/frames'

# Create dataset and dataloader
dataset = ImageDataset(img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE(h_dim=128, res_h_dim=32, n_res_layers=2,
              n_embeddings=512, embedding_dim=64, beta=0.25).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        embedding_loss, x_hat, perplexity = model(images)

        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(x_hat, images)

        # Total loss
        loss = reconstruction_loss + embedding_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, Recon Loss: {reconstruction_loss.item():.4f}, '
                  f'Embedding Loss: {embedding_loss.item():.4f}, Perplexity: {perplexity.item():.4f}')

    # Save model checkpoint
    torch.save(model.state_dict(), f'vqvae_epoch_{epoch+1}.pth')