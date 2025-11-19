import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)   # (3,32,32) -> (32,16,16)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)   # (32,16,16) -> (64,8,8)
        
        x = x.view(-1, 64 * 8 * 8)
        
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

