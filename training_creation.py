import os
import sys
import numpy as np

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from tqdm import tqdm

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_coef, iou

from skimage.morphology import label

from calculate_weight import calculate_weight

IMG_CHANNELS = 3
TRAIN_PATH = './stage1_train/'
TEST_PATH = './stage1_test/'
IMG_WIDTH = 256
IMG_HEIGHT = 256

def getting_X_Y_train():
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    weights = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    	weights[n]=calculate_weight(mask,masks, IMG_HEIGHT, IMG_WIDTH)
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    return X_train, Y_train, X_test
X_train, Y_train, X_test = getting_X_Y_train()

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
    return X_test
np.random.seed(42)
tf.random.set_seed(42)


# custom loss
def weighted_loss(y_true,y_pred):        
    log_y_pred = tf.math.log(y_pred)
    loss = tf.math.multiply(x=log_y_pred, y=weights)
    return tf.reduce_sum(loss, axis=1)


""" Hyperparaqmeters """
batch_size = 8
lr = 1e-4
num_epochs = 10
checkpoint_path = "files/test_cp.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
csv_path = "files/data.csv"

model = build_unet((IMG_HEIGHT, IMG_WIDTH, 3))
metrics = [dice_coef, iou, Recall(), Precision()]
# model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
model.compile(loss=weighted_loss, optimizer=Adam(lr), metrics=metrics)

train_steps = (len(X_train)//batch_size)
# valid_steps = (len(valid_x)//batch_size)
callbacks = [
        ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_freq=5*batch_size),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

model.fit(
        X_train, Y_train,
        epochs=num_epochs,
        # validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        # validation_steps=valid_steps,
        callbacks=callbacks
    )
model.save('model_data_science_bowl_2018.h5')
print('Done!')
