import torch

classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def test_samples(model, loader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(len(labels)):
                true_label = classes[labels[i].item()]
                pred_label = classes[preds[i].item()]
                print(f"Sample {i:2d} | True: {true_label:10s} | Pred: {pred_label}")
                
        print(f"Accuracy: {(preds == labels).sum().item() / len(outputs) * 100} %")