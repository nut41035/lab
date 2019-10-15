import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

from model import *
from data import *
import datetime


# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def combine_generator(gen1, gen2):
    while True:
        yield(next(gen1), next(gen2))

print("#### Start program")

## program parameter
TRAIN_DIR_PATH = './data/train/'
TEST_DIR_PATH = './data/test/'
seed = 1

## training parameter
EPOCHS = 10
BS = 15
IMAGE_COUNT = 5460
VALIDATION_COUNT = 4550

training_data = DataGenerator(TRAIN_DIR_PATH, batch_size=BS, image_size=64)
print('#### Successfully obtain TRAINGIN images and masks %d'%(training_data.__len__()))
testing_data = DataGenerator(TEST_DIR_PATH, batch_size=BS, image_size=64)
print('#### Successfully obtain TESTING images and masks %d'%(testing_data.__len__()))

model = unet()
print('#### Model loaded')
# model.summary()

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model.fit(training_data, 
                epochs=EPOCHS,
                validation_data=testing_data,
                callbacks=[tensorboard_callback])

model.save_weights("UNet_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
