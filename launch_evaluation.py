import torch
from pytorch_model import SimpleCNN
from torchvision import datasets, transforms
from evaluate import test_samples

def import_n_data(num_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_test_set = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # Take only 'num_data' samples
    subset_indices = list(range(num_data))
    test_subset = torch.utils.data.Subset(full_test_set, subset_indices)

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=num_data, shuffle=False)
    return test_loader


def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth"))
    
    # Choose which device the evalutation will loop on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Import data
    loader = import_n_data(600)
    
    test_samples(model, loader, device)
    
    
if __name__ == '__main__':
    main()