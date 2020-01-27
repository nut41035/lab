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

def predict(MODEL, base_path, img_name, save_to_file=True):
    image_path = os.path.join(base_path, 'images/', img_name)
    image = np.load(image_path)
    image = np.float32(image)

    image = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(image)
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction < 0.7, 0, 255)
    TP = 0
    FN = 0
    FP = 0
    if save_to_file:
        mask_path = os.path.join(base_path, 'masks/', img_name)
        mask = np.load(mask_path)
        index = int(img_name[2:6])
        red = np.array([0,0,255])
        green = np.array([0,255,0])
        blue = np.array([255,0,0])
        for i in range(32):
            img = image[0,:,:,i,:].astype(np.uint8)
            save = False
            for j in range(64):
                for k in range(64):
                    m_pixel = mask[j,k,i]
                    p_pixel = prediction[j,k,i]
                    if m_pixel == 1 or p_pixel == 255:
                        save = True
                    if m_pixel == 1 and p_pixel == 255: #TP
                        img[j,k,:] = green
                        TP+=1
                    elif m_pixel == 1 and p_pixel == 0: #FN
                        img[j,k,:] = red
                        FN+=1
                    elif m_pixel == 0 and p_pixel == 255: #FP
                        img[j,k,:] = blue
                        FP+=1
            if True:
                new_name = img_name[:-4] + "__%02d" %(index)
                # fig = plt.figure()
                # fig.suptitle(new_name, fontsize=20)
                # a = fig.add_subplot(1, 2, 1)
                # imgplot = plt.imshow(image[0,:,:,i,:].astype(np.uint8))
                # a.set_title('Original')

                # a = fig.add_subplot(1, 2, 2)
                # imgplot = plt.imshow(img)
                # a.set_title('Predictions')

                # plt.savefig('./result/%s.png'%new_name)
                # plt.close()
                cv2.imwrite('./result/%s.png'%new_name, img)
                cv2.imwrite('./result/%s.png'%img_name[:-4], image[0,:,:,i,:])
            index += 1

    return TP,FP,FN

def predict_folder(MODEL, base_path, save=True):
    files = glob.glob('%simages/*'%base_path)
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for path in tqdm(files, desc='images in folder', leave=False):
        img_name = os.path.basename(path)
        TP, FN, FP = predict(MODEL, base_path, img_name, save)
        TP_sum += TP
        FN_sum += FN
        FP_sum += FP
    return TP_sum, FP_sum, FN_sum
