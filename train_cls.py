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
from models.vqvaecls import VQVAE_cls
import os
from torchsummary import summary
import os
import torch
import argparse
from models.vqvae import VQVAE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import gc
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import torchvision.transforms as transforms

import cv2 as cv
from torchsummary import summary
parser = argparse.ArgumentParser()
from tqdm import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

model_1 = VQVAE_cls(h_dim=128, res_h_dim=32, n_res_layers=2,
              n_embeddings=512, embedding_dim=64, beta=0.25).to(device)
x = np.random.random_sample((1, 3, 128, 128))
x = torch.tensor(x).float().to(device)
print(model_1(x).shape)
# summary(model_1,(3,512,512))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_filename,model=model_1):
    path = os.getcwd() + '/'
    
    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path+model_filename,map_location=lambda storage, loc: storage)
    
    # params = data["hyperparameters"]
    
    model = model.to(device)

    model.load_state_dict(data,strict=False)
    
    return model, data
model_filename = 'vqvae_epoch_1.pth'

model,vqvae_data = load_model(model_filename)
epoch=500
batch=39
weights_dir="new_dir"
data_dir='data/final'
dataset=ImageFolder(data_dir,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]))
print(f'the class to index {dataset.class_to_idx}')
dataloader=DataLoader(dataset,batch_size=batch,shuffle=True)
model.encoder.requires_grad_(False)
model.vector_quantization.requires_grad_(False)
optimizer=optim.Adam(model.parameters(),0.001) 
epoch_loss=0
running_loss=0
num_imgs=len(dataset.imgs)
criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.1*(1-len(os.listdir(os.path.join(data_dir,file)))/num_imgs) for file in sorted(os.listdir(data_dir))],device=device))
model.train()
for i in tqdm(range(epoch),unit='epoch'):
    epoch_loss=0
    for count,data in tqdm(enumerate(dataloader),unit='iteration'):
        optimizer.zero_grad()
        model.encoder.requires_grad_(False)
        model.vector_quantization.requires_grad_(False)
        images=data[0]
        # print(f'image shape is :{images.shape}')
        label=data[1]
        # print(f'label tensor is {label}')
        # print(f'label shape is :{label.shape}')
        images=images.to(device)
        label=label.to(device)
        output=model(images)
        
        l=criterion(output,label)
      
        epoch_loss+=l.item()
        # if count%50==0:
        
        print(f'\n[ epoch :{i} iteration  :{count} loss  :{l}]')
        l.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f'{name}: {param.grad.abs().mean().item()}')

    running_loss+=epoch_loss
    
    print(f'[epoch loss:{epoch_loss}   running loss:{running_loss/(i+1)}]\n')
    torch.save(model.state_dict(),f'{weights_dir}/custom_{i}_loss:{epoch_loss}')