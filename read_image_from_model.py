from keras.models import load_model

from metrics import dice_coef
from metrics import iou

from Get_test_data import getting_X_test

import numpy as np

from skimage.transform import resize

import random
from skimage.io import imread, imshow, imread_collection, concatenate_images

import matplotlib.pyplot as plt

model = load_model('./model_data_science_bowl_2018.h5', custom_objects={'dice_coef': dice_coef, 'iou':iou})
X_test, sizes_test = getting_X_test()

idx = np.random.randint(0, len(X_test))
X_test[idx].shape
imshow(X_test[idx])

images_list = []
images_list.append(np.array(X_test[idx]))
x = np.asarray(images_list)
pr_mask = model.predict(x).round()

plt.imshow(
    pr_mask[0]
)
plt.show()
