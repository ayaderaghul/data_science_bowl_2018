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
preds_test = model.predict(X_test, verbose = 1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))

ix = random.randint(0, len(preds_test_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()