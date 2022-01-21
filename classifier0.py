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

# Plot results with matplotlib
def plot ():
    plt.figure(2)
    plt.clf()
    episode_rewards_plot = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Rewards')
    plt.plot(episode_rewards_plot.numpy())
    plt.plot(average_rewards_plot.numpy())
    plt.plot(mean_rewards_plot.numpy())
    # Take x episode averages and plot them too
    # if len(episode_rewards_plot) >= 100:
    #     means = episode_rewards_plot.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.pause(0.001)

# Plot with tensorboard
writer = SummaryWriter()

# Hyper parameters
learning_rate = 0.00001
batch_size = 16
epochs = 10

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

        # Plot with tensorboard
        writer.add_scalar("Loss/train", loss, epoch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            now = round(time.time() - start)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  time: {now}")
    print("Training done!")

# Define testing function
def test(dataloader, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct,= 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNN10(output_dim=9).to(device)
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

    data = Painting('train_info.csv', '/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed', column=4, min_paint=300, set_index=1, transform=transform)

    # Split into training set and testing set
    train_size = round(data.__len__() * 0.8)
    test_size = data.__len__() - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    print(len(train_data), len(test_data))

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
        train(train_dataloader, epochs)
        test(test_dataloader, epochs)
        writer.flush()
        # Save model
        torch.save(model.state_dict(), "/mnt/OASYS/WildfireShinyTest/CSCI364/model.pth")
        print("Saved PyTorch Model State to model.pth")
    print("Done!")

    # Save model
    torch.save(model.state_dict(), "/mnt/OASYS/WildfireShinyTest/CSCI364/model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Close writer
    writer.close()
