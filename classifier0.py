"""
Class to train the classifier
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from painting import Painting
from cnn0 import CNN0
import time

"""
Define hyper-parameters
"""
learning_rate = 1e-3
batch_size = 10
epochs = 1


"""
Define training and testing functions
"""
def train(dataloader, model, loss_fn, optimizer):
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

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    """
    Use a CNN model
    """
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNN0(output_dim=136).to(device)
    print(model)

    """
    Define loss function and optimizer
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """
    Load datasets
    """
    data = Painting('train_info.csv', 'preprocessed_1', set_index=1)
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
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in train_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
        print("Shape of y: ", y.dtype, y.shape)
        break

    """
    Train and test
    """
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    """
    Save model
    """
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")