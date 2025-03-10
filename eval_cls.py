import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from torchvision.datasets import ImageFolder

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

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix

# Main function for evaluation
def main():
    # Define the model
    model = VQVAE_cls(h_dim=128, res_h_dim=32, n_res_layers=2,
                      n_embeddings=512, embedding_dim=64, beta=0.25).to(device)
    
    # Load the trained model weights
    model_filename = 'vqvae_epoch_1.pth'
    model = load_model(model_filename, model)
    
    # Define the dataset and dataloader
    data_dir = 'data/final'  # Path to your validation/test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
    avg_loss, accuracy, precision, recall, f1, conf_matrix = evaluate(model, dataloader, criterion)
    
    # Print evaluation results
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

if __name__ == '__main__':
    main()