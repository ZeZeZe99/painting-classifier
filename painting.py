"""
Customized dataset class
"""
import os
from os.path import exists

import PIL.Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class Painting(Dataset):
    # class variables
    style = []
    genre = []

    """
    Initialize the painting dataset

    :param string annotation_file: path to the csv file, columns are: filename, artist, title, style, genre, date
    :param string img_dir: path to the image directory
    :param int min_paint: minimum number of paintings a style / genre must contain to be selected
    :param int set_index: the index of the dataset we want to use
    :param transform
    :param target_transform
    """

    def __init__(self, annotation_file, img_dir, min_paint=0, max_paint=0, set_index=0, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # for testing purpose: only need labels for 1 training set
        if set_index != 0:
            self.img_labels = self.img_labels[self.img_labels.filename.str.startswith(str(set_index))]

        # select styles that have at least x paintings, store it as a list
        if len(Painting.style) == 0:
            style_count = self.img_labels['style'].value_counts()
            if min_paint != 0:
                style_count = style_count[lambda x: x >= min_paint]
                style_count = style_count[lambda y: y <= max_paint]
            for name, item in style_count.items():
                Painting.style.append(name)
            # print(Painting.style)
            print(f"Styles selected: {len(Painting.style)}")

        # create the style and genre map, map each string to an integer
        # if len(Painting.style) == 0:
        #     Painting.style = self.img_labels['style'].unique().tolist()
        #     print("Total style: %d" % len(Painting.style))
        # if len(Painting.genre) == 0:
        #     Painting.genre = self.img_labels['genre'].unique().tolist()
        #     print("Total genre: %d" % len(Painting.genre))

        # check if the painting has the selected styles
        selected = self.img_labels.apply(self.style_selected, axis=1)
        self.img_labels = self.img_labels[selected]

        # check if a file exists (since we've removed some broken files during preprocess)
        # remove the row if the file doesn't exist
        file_exist = self.img_labels.apply(self.file_exist, axis=1)
        self.img_labels = self.img_labels[file_exist]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        # this reads in an image as uint8 tensor: RGB are in [0, 255]
        # image = read_image(img_path)

        # use this since latest version of torchvision is not available on M1 Mac
        image = PIL.Image.open(img_path)
        transform = transforms.ToTensor()
        image_tensor = transform(image)

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
        # image = image.float()

        return image_tensor, label

    """
    Check whether the file exists
    :param row: a row in the dataframe
    """

    def file_exist(self, row):
        return exists(os.path.join(self.img_dir, row['filename']))

    """
    Check whether the painting is among the selected styles
    """

    def style_selected(self, row):
        return row['style'] in Painting.style
