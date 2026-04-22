# Vibe-coded by Gemini
# Trains a CNN for euro_sat
# References:
#   https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#   https://huggingface.co/docs/datasets/quickstart#vision

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

from datasets import load_dataset

train_transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def apply_train_transforms(examples):
    examples["pixel_values"] = [train_transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def apply_test_transforms(examples):
    examples["pixel_values"] = [test_transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["label"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    print ('ResNet-18 for EuroSAT')
    print ('---------------------')

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device\n")

    print('Loading Dataset.')
    dataset = load_dataset("nielsr/eurosat-demo")['train']
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train'].with_transform(apply_train_transforms)
    test_dataset = dataset['test'].with_transform(apply_test_transforms)

    batch_size = 64
    num_workers = 2
    
    # Added pin_memory=True for faster CPU to GPU data transfer
    pin_memory = True if device != "cpu" else False
    
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print('Loading net.')
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)
    net = net.to(device)

    print('Loading loss function, optimizer, and scheduler.')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    
    num_epochs = 30 
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print('Training.')
    best_val_acc = 0.0
    PATH = './best_cnn.pth'
    
    # Early Stopping tracking variables
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_one_epoch(train_dataloader, device, optimizer, net, criterion, epoch)
        val_loss, val_acc = test_model(test_dataloader, device, net, criterion)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            print(f'*** Validation Accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%. Saving model... ***\n')
            best_val_acc = val_acc
            torch.save(net.state_dict(), PATH)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f'Validation Accuracy did not improve (Best: {best_val_acc:.2f}%). Early stopping counter: {epochs_without_improvement}/{patience}\n')
            
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered! Model hasn't improved in {patience} epochs.")
                break

    print(f'Finished Training. Best Validation Accuracy: {best_val_acc:.2f}%')


def train_one_epoch(dataloader, device, optimizer, net, criterion, epoch):
    net.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader, 0):
        inputs, labels = batch["pixel_values"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print(f'[Epoch {epoch + 1}, Batch {i + 1:3d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0


def test_model(testloader, device, net, criterion):
    net.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            outputs = net(images)
            
            # Track validation loss
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_val_loss = running_val_loss / len(testloader)
    
    print(f'Validation - Loss: {avg_val_loss:.3f}, Accuracy: {accuracy:.2f} %')
    return avg_val_loss, accuracy


if __name__ == '__main__':
    main()