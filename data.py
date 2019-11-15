import tensorflow as tf
from tensorflow import keras

import numpy as np
import cv2
import os

class DataGenerator(keras.utils.Sequence):
    # x_set is list of path to the images
    # y_set are the associated classes.

    def __init__(self, path, image_size, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.ids = [f for f in os.listdir(os.path.join(path, 'images'))]

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __load__(self, img):
        image_path = os.path.join(self.path, 'images/', img)
        mask_path = os.path.join(self.path, 'masks/', img[:-4]) + '.png'
        image = cv2.imread(image_path, 1)
        mask = cv2.imread(mask_path, 1)
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        return image, mask
        
    def __getitem__(self, idx):
        files_batch = self.ids[idx*self.batch_size : (idx+1)*self.batch_size]
        image = []
        mask = []
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)

        return image, mask
           
   
