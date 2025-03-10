import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from models.vqvae import VQVAE


# Define VQVAE model (assuming the VQVAE class is already defined)


# Custom dataset for images (for inference)
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_files[idx]  # Return image and filename

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE(h_dim=128, res_h_dim=32, n_res_layers=2,
              n_embeddings=512, embedding_dim=64, beta=0.25).to(device)

# Load the saved model weights
model_path = 'vqvae_epoch_1.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Path to the directory containing images for inference
img_dir = '/home/gunmay/frames'

# Create dataset and dataloader for inference
dataset = ImageDataset(img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Output directory to save reconstructed images
output_dir = 'reconstructed_images'
os.makedirs(output_dir, exist_ok=True)

# Inference loop
with torch.no_grad():  # Disable gradient computation
    for images, filenames in dataloader:
        images = images.to(device)

        # Forward pass (reconstruct the image)
        _,reconstructed_images,_ = model(images)

        # Denormalize the images (convert from [-1, 1] to [0, 1])
        reconstructed_images = (reconstructed_images + 1) / 2

        # Save the reconstructed images
        for i, filename in enumerate(filenames):
            reconstructed_image = reconstructed_images[i].cpu()  # Move to CPU
            reconstructed_image = transforms.ToPILImage()(reconstructed_image)  # Convert to PIL image
            output_path = os.path.join(output_dir, f'reconstructed_{filename}')
            reconstructed_image.save(output_path)
            print(f'Saved reconstructed image: {output_path}')