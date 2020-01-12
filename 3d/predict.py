import tensorflow as tf
from tensorflow import keras

import os
import cv2 as cv
import glob
import math
from imageio import imwrite
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from data import *

"""
    1. predict_img
    2. predict_folder
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def predict(MODEL, base_path, img_name, save=True):
    image_path = os.path.join(base_path, 'images/', img_name)
    image = np.load(image_path)
    image = np.float32(image)

    image = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(image)
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction < 0.7, 0, 255)
    
    if save:
        mask_path = os.path.join(base_path, 'masks/', img_name)
        mask = np.load(mask_path)
        index = int(img_name[2:6])
        for i in range(32):
            fig = plt.figure()
            a = fig.add_subplot(1, 3, 1)
            imgplot = plt.imshow(image[0,:,:,i,:].astype(np.uint8))
            a.set_title('Original')

            a = fig.add_subplot(1, 3, 2)
            imgplot = plt.imshow(prediction[:,:,i])
            a.set_title('Predictions')

            a = fig.add_subplot(1, 3, 3)
            imgplot = plt.imshow(mask[:,:,i])
            a.set_title('Mask')

            new_name = img_name[:4] + "%02d" %(index) + img_name[6:]
            plt.savefig('./result/%s.png'%new_name)
            plt.close()
            index += 1

    return prediction

def predict_folder(MODEL, base_path, save=True):
    files = glob.glob('%simages/*'%base_path)
    for path in tqdm(files, desc='images in folder', leave=False):
        img_name = os.path.basename(path)
        predict(MODEL, base_path, img_name, save)