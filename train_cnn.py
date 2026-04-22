# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://huggingface.co/docs/datasets/quickstart#vision
# Trains a CNN for euro_sat using tutorials.
#   Note: This is currently as close to the examples as possible and NOT optimized.

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datasets import load_dataset


transform = transforms.Compose([
    transforms.Resize((32, 32)), # Downsize EuroSAT (64x64) to CIFAR-10 size (32x32)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    print ('Examples:\nhttps://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html\nhttps://huggingface.co/docs/datasets/quickstart#vision\n')

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    print(f"Using {device} device\n")


    print ('Loading Dataset.')
    dataset = load_dataset("nielsr/eurosat-demo")['train']
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train'].with_transform(transforms)
    test_dataset = dataset['test'].with_transform(transforms)

    batch_size = 4
    num_workers = 2
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    print ('Loading net.')
    net = Net().to(device)


    print ('Loading loss function and optimizer.')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    print ('Training.')
    print ('[`epoch`, `batch_idx`] loss: `loss`')
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(train_dataloader, device, optimizer, net, criterion, epoch)
        test_model(test_dataloader, device, net)
    print('Finished Training\n')


    print ('Saving model.')
    PATH = './cnn.pth'
    torch.save(net.state_dict(), PATH)


def train_one_epoch(dataloader, device, optimizer, net, criterion, epoch):
    net.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch["pixel_values"].to(device), batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0


def test_model(testloader, device, net):
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, batch in enumerate(testloader, 0):
            images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on {total} test images: {100 * correct / total} %')


if __name__ == '__main__':
    main()