import numpy as np
from skimage.io import imread

x = imread('../data/raw/training.tif');x_mask = imread('../data/raw/training_groundtruth.tif')
y = imread('../data/raw/testing.tif') ;y_mask = imread('../data/raw/testing_groundtruth.tif')

for slice in range(165):
    data = np.concatenate([x[[slice]],x_mask[[slice]]])
    np.save('./data/training/' + f'data_{100+slice}.npy',data)
    data = np.concatenate([y[[slice]],y_mask[[slice]]])
    np.save('./data/testing/' + f'data_{100+slice}.npy',data)