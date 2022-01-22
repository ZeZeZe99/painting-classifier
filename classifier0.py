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
from cnn7 import CNN7
from cnn10 import CNN10
from znn import ZNN
from znn2 import ZNN2

# Hyper parameters
learning_rate = 0.0008
batch_size = 16
epochs = 500


# Define training function
def train(dataloader, epoch):
    # Initialize
    start = time.time()
    size = len(dataloader.dataset)

    model.train()
    loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Plot to tenserboard
        writer.add_scalar("train/loss", loss, batch + len(dataloader) * epoch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")

    print(f"Epoch {epoch + 1} training done!\n")


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
    print(f"Valid accuracy: {(100 * accuracy):>0.1f}%, Valid avg loss: {avg_loss:>8f}")
    print(f"Epoch {epoch + 1} validation done!\n")

    # Plot with tensorboard
    writer.add_scalar("valid/loss", avg_loss, epoch)
    writer.add_scalar("valid/accuracy", 100 * accuracy, epoch)


# Define testing function
def test(dataloader):
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
    print(f"Test accuracy: {(100 * accuracy):>0.1f}%, Test avg loss: {avg_loss:>8f} \n")


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = ZNN2(output_dim=9).to(device)
    # model = models.resnet18(pretrained=True).to(device)
    print(model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomRotation(25),
        # transforms.Resize(224) # if resnet
    ])

    # data = Painting('train_info.csv', 'preprocessed_1', column=4, min_paint=300, set_index=1, transform=transform)
    # data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed', column=4, min_paint=300,
    #                 set_index=1, transform=transform)
    data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed', column=1, min_paint=390, set_index=0, transform=transform)

    # Split into training set, validation set, and testing set
    test_size = round(data.__len__() * 0.15)
    validate_size = test_size
    train_size = data.__len__() - 2 * test_size
    train_data, validate_data, test_data = random_split(data, [train_size, validate_size, test_size])
    print(len(train_data), len(validate_data), len(test_data))

    # Create dataset loaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for X, y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
        print("Shape of y: ", y.dtype, y.shape)
        break

    # Plot with tensorboard
    writer = SummaryWriter()

    # Train and test
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        torch.cuda.empty_cache()
        train(train_dataloader, t)
        validate(validate_dataloader, t)
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
