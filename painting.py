"""
Customized dataset class
"""
import os
from os.path import exists

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class Painting(Dataset):
    # class variables
    style = []
    genre = []

    """
    Initialize the painting dataset

    :param annotation_file: path to the csv file, columns are: filename, artist, title, style, genre, date
    :param img_dir: path to the image directory
    :param set_index: the index of the dataset we want to use
    :param transform
    :param target_transform
    """
    def __init__(self, annotation_file, img_dir, set_index=0, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # create the style and genre map
        if len(Painting.style) == 0:
            Painting.style = self.img_labels['style'].unique().tolist()
            print("Total style: %d" % len(Painting.style))
        if len(Painting.genre) == 0:
            Painting.genre = self.img_labels['genre'].unique().tolist()
            print("Total genre: %d" % len(Painting.genre))

        # for testing purpose: only need labels for 1 training set
        if set_index != 0:
            self.img_labels = self.img_labels[self.img_labels.filename.str.startswith(str(set_index))]

        # check if a file exists (since we've removed some broken files during preprocess)
        # remove the row if the file doesn't exist
        file_exist = self.img_labels.apply(self.file_exist, axis=1)
        self.img_labels = self.img_labels[file_exist]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # this reads in an image as uint8 tensor: RGB are in [0, 255]
        image = read_image(img_path)

        # convert the label from string to integer
        # use style as the label
        label = self.img_labels.iloc[idx, 3]
        label = Painting.style.index(label)
        # use genre as the label
        # label = self.img_labels.iloc[idx, 4]
        # label = Painting.genre.index(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # convert the image tensor to float: RGB are in [0, 1]
        image = image.float()

        return image, label

    """
    Check whether the file exists
    :param row: a row in the dataframe
    """
    def file_exist(self, row):
        return exists(os.path.join(self.img_dir, row['filename']))