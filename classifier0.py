# Classifier
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

from painting import Painting
from cnn0 import CNN0
from cnn1 import CNN1
from cnn2 import CNN2
from cnn3 import CNN3
from cnn4 import CNN4
from cnn5 import CNN5
from cnn7 import CNN7
from cnn8 import CNN8
from cnn9 import CNN9
from cnn10 import CNN10

# Plot with tensorboard
writer = SummaryWriter()

# Hyper parameters
learning_rate = 0.00001
batch_size = 16
epochs = 5

# Define training function
def train(dataloader, epoch):
    # Initialize
    start = time.time()
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")

    # Plot with tensorboard
    print(f"Epoch {epoch} training done!")

# Define validation function
def validate(dataloader, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            valid_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = valid_loss / num_batches
    accuracy = correct / size
    print(f"Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    print(f"Epoch {epoch} validation done!")


# Define testing function
def test(dataloader, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    writer.add_scalar("Loss/train", test_loss, epoch)
    print(f"Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNN0(output_dim=9).to(device)
    # model = models.resnet18(pretrained=True).to(device)
    print(model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(32) # if resnet50
    ])

    # data = Painting('train_info.csv', 'preprocessed_1', column=4, min_paint=300, set_index=1, transform=transform)
    data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed', column=4, min_paint=300, set_index=1, transform=transform)

    # Split into training set, validation set, and testing set
    test_size = round(data.__len__() * 0.15)
    validate_size = test_size
    train_size = data.__len__() - 2 * test_size
    train_data, validate_data, test_data = random_split(data, [train_size, validate_size, test_size])
    print(len(train_data), len(validate_data), len(test_data))

    # Create dataset loaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
        print("Shape of y: ", y.dtype, y.shape)
        break

    # Train and test
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        torch.cuda.empty_cache()
        train(train_dataloader, t)
        validate(test_dataloader, t)
        # Save model
        torch.save(model.state_dict(), "/mnt/OASYS/WildfireShinyTest/CSCI364/model.pth")
        print("Saved PyTorch Model State to model.pth")
    print("Training and validation done! Testing start ------------------")
    test(test_dataloader)
    print("Done!")

    # Save model
    torch.save(model.state_dict(), "/mnt/OASYS/WildfireShinyTest/CSCI364/model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Close writer
    writer.flush()
    writer.close()
