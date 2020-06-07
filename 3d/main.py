import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import numpy as np
import os
import cv2 as cv
# import matplotlib.pyplot as plt

from model import *
from data import *
from loss import *
from predict import *
import datetime
import requests
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto, GPUOptions, InteractiveSession

def train_and_eval(EPOCHS = 20, BS = 16, IMAGE_COUNT = 139, VALIDATION_COUNT =134, learning_rate = 0.05, beta = 0.5):
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    gpu_options = GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("#### Start program")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    ## program parameter
    BASE_DIR = './data/processed/3-channel-1/'
    TRAIN_DIR_PATH = BASE_DIR + 'train/'
    VALIDATION_DIR_PATH = BASE_DIR + 'validation/'
    seed = 1

    ## training parameter
    # EPOCHS = 20
    BS = 16
    # IMAGE_COUNT = 470
    # VALIDATION_COUNT =134
    # learning_rate = 0.005
    # beta = 0.4
    # loss_func = weighted_dice_loss(beta)
    loss_func = tversky_loss(beta)
    
    # loss_func = dice_coef_loss
    # more mean no FP
    input_size = (64,64,32,3)

    training_data = DataGenerator(TRAIN_DIR_PATH, batch_size=BS, image_size=64)
    print('#### Successfully obtain TRAINGIN images and masks %d'%(training_data.__len__()))
    validating_data = DataGenerator(VALIDATION_DIR_PATH, batch_size=BS, image_size=64)
    print('#### Successfully obtain VALIDATING images and masks %d'%(validating_data.__len__()))

    model = unet_norm(input_size = input_size,loss_func=loss_func,l_rate=learning_rate)
    print('#### Model loaded')
    # model.summary()

    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir="logs/fit/" + time_stamp
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    file_writer = tf.summary.create_file_writer(log_dir)
    # model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

    model.fit_generator(training_data,
                    steps_per_epoch=100,
                    epochs=EPOCHS,
                    validation_data=validating_data,
                    callbacks=[tensorboard_callback])

    model.save("model/UNet_%s.h5" %time_stamp)
    print("model saved at   model/UNet_%s.h5"%time_stamp)

    text = 'UNet_%s.h5\r\
            loss: weighted_dice  %s\r\
            learninf rate: %s\r\
            image size: %s\r\
            comment: training on seperate dataset\r'\
            %(time_stamp, beta,learning_rate,input_size)

    with open("./log.txt", "a") as myfile:
        myfile.write(text)

    # TP_sum, FP_sum, FN_sum = predict_folder(model, '%stest/'%BASE_DIR, save_mode=3, save_dir='./result/%s'%(time_stamp))
    # text = 'Evaluation result\r\
    #         TP : %s\r\
    #         FP : %s\r\
    #         FN : %s\r\r'\
    #         %(TP_sum, FP_sum, FN_sum)
    # with open("./log.txt", "a") as myfile:
    #     myfile.write(text)

    InteractiveSession.close(session)
    # predict_folder(model, BASE_DIR)
    # predict_whole_body(model, 'data/raw/case1/', 'DWIBS AI CASE 1.0037', 128, 128)

if __name__ == "__main__":
    train_and_eval()