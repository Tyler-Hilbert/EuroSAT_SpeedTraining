# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://huggingface.co/docs/datasets/quickstart#vision

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datasets import load_dataset


transform = transforms.Compose([
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
        # Changed 16 * 5 * 5 to 16 * 13 * 13
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
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
    dataset = dataset.with_transform(transforms)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )


    print ('Loading net.')
    net = Net().to(device)


    print ('Loading loss function and optimizer.')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    print ('Training.')
    print ('[`epoch`, `batch_idx`] loss: `loss`')
    for epoch in range(10):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training\n')


    print ('Saving model.')
    PATH = './cnn.pth'
    torch.save(net.state_dict(), PATH)


    # FIXME train/test split and check results
    print ('\nFIXME - create train/test split and check results.')

if __name__ == '__main__':
    main()