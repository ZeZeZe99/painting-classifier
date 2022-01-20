"""
This program is used to preprocess images to 256*256 jpg format.
"""

import os
import sys

import PIL
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg


# display images
plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


# transformation: first resize the smaller dimension to 256, then crop the center 256*256 square
transforms = nn.Sequential(
    # T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(256)
)

if __name__ == '__main__':

    # get input and output directory from keyboard input
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # get filenames
    img_names = os.listdir(input_path)

    count = 0
    success = 0

    for i in range(len(img_names)):
        filename = img_names[i]
        try:
            # read image using torchvision
            # img = read_image(input_path + '/' + filename)
            # only keep images with 3 channels
            # if img.shape[0] != 3:
            #     continue

            # read image using PIL
            img = PIL.Image.open(input_path + '/' + filename)
            # only keep images with 3 channels
            if img.mode != "RGB":
                continue


            # perform resizing and cropping
            t_img = transforms(img)

            # save the output jpeg file to a different directory with the same name
            # using torchvision
            # write_jpeg(t_img, output_path + '/' + filename)

            # using PIL
            t_img.save(output_path + '/' + filename)
            success += 1
            if success % 100 == 0:
                print(success)

        except Exception as e:
            print(filename)
            print(e)
            count += 1
    print('Failure: %d' % count)
