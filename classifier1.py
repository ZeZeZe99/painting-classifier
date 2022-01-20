import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from painting import Painting
from cnn5 import CNN5

# Normalization
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means,std)])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means,std)])

# Hyper parameters
num_epochs = 35
num_classes = 2
batch_size = 25
learning_rate = 0.001

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

