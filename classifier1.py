import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time

from painting import Painting
from cnn0 import CNN0
from cnn5 import CNN5

# Normalization
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# test_transform = transforms.Compose([transforms.ToPILImage(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(means,std)])
#
# valid_transform = transforms.Compose([transforms.ToPILImage(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(means,std)])

# Hyper parameters
epochs = 35
num_classes = 2
batch_size = 25
learning_rate = 0.001

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define training and testing functions
def train(dataloader):
    start = time.time()
    size = len(dataloader.dataset)

    # Keep track of loss
    train_loss = 0.0
    valid_loss = 0.0

    # Training
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Move tensors to GPU
        data, target = X.to(device), y.to(device)

        # Clear the gradients of all optimized var
        optimizer.zero_grad()

        # Forward pass, compute predicted outputs
        output = model(data)

        # Calculate the batch loss
        loss = criterion(output, target)

        #print("Label:   ", target)
        #print("Predict: ", output.argmax(1))

        # Backpropagation, compute gradient of the loss
        loss.backward()
        # Update parameters
        optimizer.step()

        # Update training loss
        # train_loss += loss.item() * data.size(0)

        if batch % 50 == 0:
            train_loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")
    print("Training done!")


def test(dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            images, labels = X.to(device), y.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # test_loss += loss_fn(pred, y).item()
            print("guess: ", predicted)
            print("label: ", labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNN5().to(device)
    print(model)

    # Define loss and optimizer function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """
    Load datasets
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224)
    ])

    data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed_1', min_paint=1000, max_paint=1300,
                    set_index=1, transform=train_transforms)
    # data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed_1', min_paint=1000, set_index=0, transform= train_transforms)
    # print(train_data.__len__())

    """
    Split into training set and testing set
    """
    train_size = round(data.__len__() * 0.8)
    test_size = data.__len__() - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    print(len(train_data), len(test_data))

    """
    Create dataset loaders
    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for X, y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
        print("Shape of y: ", y.dtype, y.shape)
        break

    """
    Train and test
    """
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader)
        test(test_dataloader)
    print("Done!")

    """
    Save model
    """
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")