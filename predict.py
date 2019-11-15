import tensorflow as tf
from tensorflow import keras

import os
import cv2 as cv
import glob
import math
from imageio import imwrite
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data import *

def test(BASE_DIR, MODEL_VERSION):
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    print("#### Start program")

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if not os.path.exists('%sresult'%BASE_DIR):
        os.makedirs('%sresult'%BASE_DIR)
    if not os.path.exists('%sresult/predicted'%BASE_DIR):
        os.makedirs('%sresult/predicted'%BASE_DIR)
    if not os.path.exists('%sresult/combined'%BASE_DIR):
        os.makedirs('%sresult/combined'%BASE_DIR)

    TEST_DIR_PATH = BASE_DIR + 'test/images/'
    files = glob.glob('%s/*'%TEST_DIR_PATH)

    model = keras.models.load_model('model/%s.h5'%MODEL_VERSION)
    print('#### Model loaded')
    model.summary()

    for path in files:
        imgName = os.path.basename(path)[0:-4]
        print(imgName)
        image = cv.imread(path, 1)
        image = np.array([image/255.0])

        predictions = model.predict(image)
        predictions = np.squeeze(predictions)

        imwrite('%s/result/predicted/%s.png'%(BASE_DIR, imgName), predictions)

        fig = plt.figure()
        a = fig.add_subplot(1, 3, 1)
        img =  cv.imread(path, 1)
        imgplot = plt.imshow(img)
        a.set_title('Original')

        a = fig.add_subplot(1, 3, 2)
        imgplot = plt.imshow(predictions)
        a.set_title('Predictions')

        a = fig.add_subplot(1, 3, 3)
        print('%smasks/%s.png'%(BASE_DIR, imgName))
        img =  cv.imread('%stest/masks/%s.png'%(BASE_DIR, imgName), 1)
        imgplot = plt.imshow(img)
        a.set_title('Mask')

        plt.savefig('%sresult/combined/%s.png'%(BASE_DIR, imgName))

def whole_body_test(image_path, image_name, MODEL_VERSION, h = 64, w = 64):
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    print("#### Start program")

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if not os.path.exists('./result'):
        os.makedirs('./result')

    model = keras.models.load_model('model/%s.h5'%MODEL_VERSION)
    print('#### Model loaded')

    print(image_name)
    path = image_path + 'images/' + image_name + '.jpg'
    image = cv.imread(path, 1)
    mask_path = image_path + 'masks/' + image_name + '.jpg'
    mask = cv.imread(mask_path, 1)

    height = image.shape[0]
    width = image.shape[1] 
    hloop = int(math.floor(width/w)) #Horizontal loop
    vloop = int(math.floor(height/h)) #Vertical loop
    result = np.array([[0.0 for x in range(width)] for y in range(height)])
    for i in range(hloop):
        for j in range(vloop):
            crop_img = image[h*j:h*(j+1), w*i:w*(i+1)]
            crop_img = np.array([crop_img/255.0])
            predictions = model.predict(crop_img)
            predictions = np.squeeze(predictions)

            result[h*j:h*(j+1), w*i:w*(i+1)] = predictions
    imwrite('./result/%s_predicted.png'%image_name, result)
              
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(image)
    a.set_title('Original')

    a = fig.add_subplot(1, 3, 2)
    result = cv.imread('./result/%s_predicted.png'%image_name, 1)
    imgplot = plt.imshow(result)
    a.set_title('Predictions')

    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(mask)
    print(mask.shape)
    a.set_title('Mask')

   
    plt.savefig('./result/%s_combined.png'%image_name)