import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image, write_jpeg
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader


plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


transforms = nn.Sequential(
    T.Resize(256),
    T.CenterCrop(256)
)

if __name__ == '__main__':
    # get filenames
    path = 'train_2/'
    img_names = os.listdir(path)
    for i in range(len(img_names)):
        filename = img_names[i]
        try:
            img = read_image(path + filename)
            t_img = transforms(img)

            t_img = T.ToPILImage()(t_img)

            if t_img.mode != 'RGB' and t_img.mode != 'L':
                t_img = t_img.convert("RGB")
            t_img.save('processed/' + filename)
        except:
            print(filename)




    # img = read_image(path + '2.jpg')
    # show([img1,t_img1, t_img2, t_img3])
    #
    #
    # # write_jpeg(t_img1, 'processed/' + filename)
    # save_image(t_img1, 'proccessed/' + filename)
    # # torch.save(t_img1, 'proccessed/'+filename)
    # # Image.fromarray(t_img1.numpy()).save('a.jpg')
    # img2 = read_image('proccessed/'+filename)
