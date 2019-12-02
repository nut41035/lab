import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import numpy as np
import os
import cv2 as cv
# import matplotlib.pyplot as plt

from model import *
from data import *
from predict import *
import datetime
from notifiers import get_notifier

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print("#### Start program")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

## program parameter
BASE_DIR = './data/processed/only_cancers/'
TRAIN_DIR_PATH = BASE_DIR + 'train/'
VALIDATION_DIR_PATH = BASE_DIR + 'validation/'
seed = 1

## training parameter
EPOCHS = 20
BS = 20
IMAGE_COUNT = 528
VALIDATION_COUNT = 63

training_data = DataGenerator(TRAIN_DIR_PATH, batch_size=BS, image_size=64)
print('#### Successfully obtain TRAINGIN images and masks %d'%(training_data.__len__()))
validating_data = DataGenerator(VALIDATION_DIR_PATH, batch_size=BS, image_size=64)
print('#### Successfully obtain VALIDATING images and masks %d'%(validating_data.__len__()))

model = unet(input_size = (64,64,3))
print('#### Model loaded')
# model.summary()

time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir="logs/fit/" + time_stamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model.fit(training_data, 
                epochs=EPOCHS,
                validation_data=validating_data,
                callbacks=[tensorboard_callback])

model.save("model/UNet_%s.h5" %time_stamp)
print("model saved at   model/UNet_%s.h5"%time_stamp)

predict_folder(model, BASE_DIR)
predict_whole_body(model, 'data/raw/case1/', 'DWIBS AI CASE 1.0037', 64, 64)

p = get_notifier('slack')
p.notify(webhook_url='https://hooks.slack.com/services/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', 
         message='Training complete model    UNet_%s.h5'%time_stamp)
