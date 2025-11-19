import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from evaluate import evaluate

def training_loop(model, train_loader, test_loader, device):
    
    # Set hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 5
    
    pbar = tqdm(total=len(train_loader))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar.set_description(f"Epoch {epoch}: ")
        count = 0

        for images, labels in train_loader:
            count += 1
            pbar.n = count
            pbar.refresh()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward propagation
            predictions = model(images)
            
            # Compute the loss function
            loss = criterion(predictions, labels)
            
            # Backward propagation (set the gradients)
            loss.backward()
            
            # Apply the gradients
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
        evaluate(model, test_loader, device)

    pbar.close()
