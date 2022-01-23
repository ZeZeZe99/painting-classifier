# ZBNet for Painting Classification

Because the image data we have is too big, we currently store them on the lab computer at:

/mnt/OASYS/WildfireShinyTest/CSCI364/preprocessed

## Program

`preprocess`: Preprocess the image files to 256\*256

`paint_stat.py`: Print out statistics of selected image folders

`painting.py`: Customized dataset class to organized preprocessed images and labels for dataloader

`classifier.py`: The main program that runs the training, validation and testing. Each run will be saved into ./runs, where we kept a history of each run data for analysis and comparision.

`ZBNet.py`: Our implementation of neural network

`cnn_inception.py`: The inception module for ZBNet

`cn_residual.py`: The residual net module for neural network. We did not use it in the end, because the performance is not improved.

`/runs`: We storce all our run data here for Tensorboard data visualization.

## Tensorboard

To visualize all our run data, install Tnensorboard and run the following command:

`tensorboard --logdir=runs`
