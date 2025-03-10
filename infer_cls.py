import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import argparse
from models.vqvaecls import VQVAE_cls
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
def load_model(model_filename, model):
    path = os.getcwd() + '/'
    
    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path + model_filename, map_location=lambda storage, loc: storage)
    
    model = model.to(device)
    model.load_state_dict(data, strict=False)
    
    return model

# Inference function
def infer(model, image_path, transform):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Forward pass through the model
        output = model(image)
        
        # Get the predicted class
        _, predicted = torch.max(output, 1)
        
        return predicted.item()

# Main function for inference
def main():
    # Define the model
    model = VQVAE_cls(h_dim=128, res_h_dim=32, n_res_layers=2,
                      n_embeddings=512, embedding_dim=64, beta=0.25).to(device)
    
    # Load the trained model weights
    model_filename = 'vqvae_epoch_1.pth'
    model = load_model(model_filename, model)
    
    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Path to the image you want to infer
    image_path = 'path_to_your_image.jpg'
    
    # Perform inference
    predicted_class = infer(model, image_path, transform)
    
    # Print the predicted class
    print(f'Predicted class: {predicted_class}')

if __name__ == '__main__':
    main()