import torch
import torch.nn as nn
from torchsummary import summary

import torch
import torch.nn as nn
from torchsummary import summary

class ClassificationHead(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassificationHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Corrected input size for fc1
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Input size = 4096
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # print(x.shape)
        return x
if __name__ == "__main__":
    device = torch.device('cpu')  
    model = ClassificationHead(num_classes=2).to(device)
    print(model)
    input_tensor = torch.randn(2, 64, 32, 32).to(device)  
    output = model(input_tensor)
    print(output.shape)  
    summary(model, (64, 32, 32),device='cpu')