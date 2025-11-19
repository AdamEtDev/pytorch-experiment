import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_model import SimpleCNN
from training import training_loop
import matplotlib.pyplot as plt

def import_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR 10 dataset = 60 000 images 32x32
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # 50 000 data
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # 10 000 data

    # Split the data into batches
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def main():
    model = SimpleCNN()
    
    # Choose which device the training will loop on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Importing data
    train_loader, test_loader = import_data()
    
    # Launch training
    training_loop(model, train_loader, test_loader, device)
    
    # Saving model
    torch.save(model.state_dict(), "model.pth")
    

if __name__ == '__main__':
    main()