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

from data import *

"""
    1. predict_img
    2. predict_path
    3. predict_folder
    4. predict_whole_body
"""

def predict_img(MODEL, img):
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    image = np.array([img/255.0])
    prediction = MODEL.predict(image)
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction < 0.7, 0, 255)
    prediction = prediction.astype(np.uint8)

    return prediction

def predict_path(MODEL, BASE_DIR, imgName): 
    imagePath = BASE_DIR + 'images/' + imgName + '.jpg'
    image = cv.imread(imagePath, 1)
    prediction = predict_img(MODEL, image)
    
    maskPath = BASE_DIR + 'masks/' + imgName + '.npy'
    mask = np.load(maskPath)

    return image, mask, prediction

def predict_folder(MODEL, BASE_DIR):
    if not os.path.exists('%sresult'%BASE_DIR):
        os.makedirs('%sresult'%BASE_DIR)
    if not os.path.exists('%sresult/predicted'%BASE_DIR):
        os.makedirs('%sresult/predicted'%BASE_DIR)
    if not os.path.exists('%sresult/combined'%BASE_DIR):
        os.makedirs('%sresult/combined'%BASE_DIR)

    TEST_DIR_PATH = BASE_DIR + 'test/'
    files = glob.glob('%s/images/*'%TEST_DIR_PATH)
    for path in tqdm(files, desc='images in folder', leave=False):
        imgName = os.path.basename(path)[0:-4]
        image, mask, prediction = predict_path(MODEL, TEST_DIR_PATH, imgName)
        imwrite('%s/result/predicted/%s.png'%(BASE_DIR, imgName), prediction)

        fig = plt.figure()
        a = fig.add_subplot(1, 3, 1)
        imgplot = plt.imshow(image)
        a.set_title('Original')

        a = fig.add_subplot(1, 3, 2)
        imgplot = plt.imshow(prediction)
        a.set_title('Predictions')

        a = fig.add_subplot(1, 3, 3)
        imgplot = plt.imshow(mask)
        a.set_title('Mask')

        plt.savefig('%sresult/combined/%s.png'%(BASE_DIR, imgName))
        plt.close()

def predict_whole_body(MODEL, image_path, image_name, h = 64, w = 64):
    if not os.path.exists('./result'):
        os.makedirs('./result')

    path = image_path + 'images/' + image_name + '.jpg'
    image = cv.imread(path, 1)
    mask_path = image_path + 'masks/' + image_name + '.jpg'
    mask = cv.imread(mask_path, 1)

    img = cv.resize(image,(1024,1024))
    height = 1024
    width = 1024
    hloop = int(math.ceil(width/w)) #Horizontal loop
    vloop = int(math.ceil(height/h)) #Vertical loop
    result = np.array([[0.0 for x in range(width)] for y in range(height)])
    for i in range(hloop):
        for j in range(vloop):
            crop_img = img[h*j:h*(j+1), w*i:w*(i+1),:]
            prediction = predict_img(MODEL, crop_img)
            result[h*j:h*(j+1), w*i:w*(i+1)] = prediction
    result = cv.resize(result,(480,865))
    result = np.where(result < 127, 0, 255)
    result = result.astype(np.uint8)
    imwrite('./result/%s_predicted.png'%image_name, result)
              
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(image)
    a.set_title('Original')

    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(result)
    a.set_title('Predictions')

    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(mask)
    a.set_title('Mask')

    plt.savefig('./result/%s_combined.png'%image_name)
    plt.close()

def predict_folder_whole_body(MODEL, image_path, h = 64, w = 64):
    files = glob.glob('%s/images/*'%image_path)
    for path in tqdm(files, desc='predicting WB image', leave=False):
        imgName = os.path.basename(path)[0:-4]
        predict_whole_body(MODEL, image_path, imgName, h, w)