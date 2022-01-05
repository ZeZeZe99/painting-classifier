"""
This program is used to preprocess images to 256*256 jpg format.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg


# display images
# plt.rcParams["savefig.bbox"] = 'tight'
# torch.manual_seed(1)
#
#
# def show(imgs):
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = T.ToPILImage()(img.to('cpu'))
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#     plt.show()


# transformation: first resize the smaller dimension to 256, then crop the center 256*256 square
transforms = nn.Sequential(
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
    for i in range(len(img_names)):
        filename = img_names[i]
        try:
            img = read_image(input_path + '/' + filename)
            # perform resizing and cropping
            t_img = transforms(img)
            # save the output jpeg file to a different directory with the same name
            write_jpeg(t_img, output_path + '/' + filename)

            # another way to save the jpeg file
            # t_img = T.ToPILImage()(t_img)
            # if t_img.mode != 'RGB' and t_img.mode != 'L':
            #     t_img = t_img.convert("RGB")
            # t_img.save('processed/' + filename)
        except Exception as e:
            print(filename)
            print(e)
            count += 1
    print('Failure: %d' % count)
