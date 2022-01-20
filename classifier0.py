"""
Class to train the classifier
"""
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from painting import Painting
from cnn0 import CNN0
from cnn1 import CNN1
from cnn2 import CNN2
from cnn3 import CNN3
from cnn4 import CNN4
import time

"""
Define hyper-parameters
"""
learning_rate = 0.001
batch_size = 16
epochs = 100


"""
Define training and testing functions
"""
def train(dataloader):
    start = time.time()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # print("Input:   ", X)
        print("Label:   ", y)
        # print("Output:  ", pred)
        print("Predict: ", pred.argmax(1))
        # print("l: ", loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")
    print("Training done!")


def test(dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    plot = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # if pred.argmax(1) == y:
            print("pred: ", pred.argmax(1))
            print("y: ", y)
            # print(pred.argmax(1) == y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    print(correct, size)
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # plot.append(100*correct)
    # acc_plot = torch.tensor(plot, dtype=torch.float)
    # plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Accuracy')
    # plt.plot(acc_plot.numpy())
    # plt.pause(0.001)


if __name__ == '__main__':
    """
    Use a CNN model
    """
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNN0(output_dim=14).to(device)
    # model = models.resnet50(pretrained=True).to(device)
    print(model)
    exit()

    """
    Define loss function and optimizer
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """
    Load datasets
    """
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224)
    ])

    data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed_1', min_paint=1000, max_paint=1300,
                    set_index=1)
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
        train(train_dataloader)
        test(test_dataloader)
    print("Done!")

    """
    Save model
    """
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")