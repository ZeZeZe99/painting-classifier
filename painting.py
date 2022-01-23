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
    selected_labels = []
    painting_count = []

    """
    Initialize the painting dataset

    :param string annotation_file: path to the csv file, columns are: filename, artist, title, style, genre, date
    :param string img_dir: path to the image directory
    :param int min_paint: minimum number of paintings a style / genre must contain to be selected
    :param int set_index: the index of the dataset we want to use
    :param transform
    :param target_transform
    """

    def __init__(self, annotation_file, img_dir, column=3, min_paint=None, max_paint=None, name_start_with=None, set_index=None,
                 transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.column = column
        self.name_start_with = name_start_with
        self.min_paint = min_paint
        self.max_paint = max_paint

        # for testing purpose: only need labels for 1 training set

        if name_start_with:
            name = self.img_labels.apply(self.name_start, axis=1)
            self.img_labels = self.img_labels[name]
        if set_index:
            self.img_labels = self.img_labels[self.img_labels.filename.str.startswith(str(set_index))]

        # use style (column index 3) as label
        if column == 3:
            if len(Painting.selected_labels) == 0:
                # select styles that have specific amount of paintings, store it as a list
                style_count = self.img_labels['style'].value_counts()

                if min_paint:
                    style_count = style_count[lambda x: x >= min_paint]
                if max_paint:
                    style_count = style_count[lambda y: y <= max_paint]

                for name, item in style_count.items():
                    Painting.selected_labels.append(name)
                print(f"Styles selected: {len(Painting.selected_labels)}")
                print(f"Styles selected: {Painting.selected_labels}")

        # use genre (column index 4) as label
        elif column == 4:
            if len(Painting.selected_labels) == 0:
                # select genres that have specific amount of paintings, store it as a list
                genre_count = self.img_labels['genre'].value_counts()
                if min_paint:
                    genre_count = genre_count[lambda x: x >= min_paint]
                if max_paint:
                    genre_count = genre_count[lambda y: y <= max_paint]
                for name, item in genre_count.items():
                    Painting.selected_labels.append(name)
                # print(Painting.style)
                print(f"Genres selected: {len(Painting.selected_labels)}")
                print(f"Genres selected: {Painting.selected_labels}")

        # use artist (column index 1) as label
        elif column == 1:
            if len(Painting.selected_labels) == 0:
                # select artists that have specific amount of paintings, store it as a list
                artist_count = self.img_labels['artist'].value_counts()
                if min_paint:
                    artist_count = artist_count[lambda x: x >= min_paint]
                if max_paint:
                    artist_count = artist_count[lambda y: y <= max_paint]
                for name, item in artist_count.items():
                    Painting.selected_labels.append(name)
                print(f"Artists selected: {len(Painting.selected_labels)}")
                print(f"Artists selected: {Painting.selected_labels}")


        # check if the label (style/genre/artist) of the painting is selected
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
        # transform = transforms.ToTensor()
        # image_tensor = transform(image)

        # convert the label from string to integer
        # use style as the label
        label = self.img_labels.iloc[idx, self.column]
        label = Painting.selected_labels.index(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # convert the image tensor to float: RGB are in [0, 1]
        # image = image.float()

        return image, label

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
        if self.column == 3:
            return row['style'] in Painting.selected_labels
        elif self.column == 4:
            return row['genre'] in Painting.selected_labels
        elif self.column == 1:
            return row['artist'] in Painting.selected_labels

    def name_start(self, row):
        return row['filename'][0] in self.name_start_with
