import os
import sys
import numpy as np

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from tqdm import tqdm
IMG_CHANNELS = 3
TEST_PATH = './stage1_test/'
IMG_WIDTH = 256
IMG_HEIGHT = 256

def getting_X_test():
    test_ids = next(os.walk(TEST_PATH))[1]
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    return X_test, sizes_test


