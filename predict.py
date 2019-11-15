import tensorflow as tf
from tensorflow import keras

import os
import cv2 as cv
import glob
from imageio import imwrite
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data import *

def test(BASE_DIR = './data/processed/only_cancers/',MODEL_VERSION = 'UNet_20191115-162049'):
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