import tensorflow as tf
from tensorflow import keras

import os
import cv2
import glob
import math
from imageio import imwrite
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from data import *
from loss import *

"""
    1. predict_img
    2. predict_folder
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

f_score_tres = 0.5
#f_score_beta  => recall is b times as important as precision
f_score_beta = 1

def predict(MODEL, base_path, img_name, save_mode=0, save_dir='./result'):
    """ save_mode : 0 = not save
                    1 = save all
                    2 = save only cancer in mask
                    3 = save only predicted something
                    4 = save only contain cancer || prediced something
                    5 = save only contain true positive
    """
    image_path = os.path.join(base_path, 'images/', img_name)
    image = np.load(image_path)
    image = np.float32(image)
    image = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(image)
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction < 0.7, 0, 255)
    # print('pred',prediction.shape)
    mask_path = os.path.join(base_path, 'masks/', img_name)
    masks = np.load(mask_path)
    masks = np.where(masks < 0.7, 0, 255)
    # print('mask',masks.shape)
    # masks = np.swapaxes(masks,2,3)
    masks = np.squeeze(masks)
    index = int(img_name[2:6])
    red = np.array([0,0,255])
    green = np.array([0,255,0])
    blue = np.array([255,0,0])
    TP_A, FP_A, FN_A = 0, 0, 0
    for i in range(32):
        img = image[0,:,:,i,:].astype(np.uint8)
        mask = masks[:,:,i].astype(np.uint8)
        mask_contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        mask_contours = imutils.grab_contours(mask_contours)

        pred = prediction[:,:,i].astype(np.uint8)
        pred_contours = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pred_contours = imutils.grab_contours(pred_contours)
        # print(len(mask_contours)+len(pred_contours))
        # not_null = False if (len(mask_contours)+len(pred_contours))==0 else True
        if save_mode == 0:
            save = False
        elif save_mode == 1:
            save = True
        elif save_mode == 2: # save only cancer in mask
            save = True if (len(mask_contours)>0) else False
        elif save_mode == 3: # save only predicted something
            save = True if (len(pred_contours)>0) else False
        elif save_mode == 4: # save only contain cancer || prediced something
            save = True if (len(pred_contours)>0 or len(mask_contours)>0) else False

        # save = False if (len(mask_contours))==0 else True
        # save=True
        # cv2.drawContours(pred, pred_contours, 1, (255), 3)
        FP_temp = np.array(pred_contours.copy())
        TP_temp = 0
        # print(np.shape(pred_contours),np.shape(mask_contours))
        # print(np.shape(FN_temp))
        # break
        for m in mask_contours:
            for p in pred_contours:
                TP,FP,FN = 0,0,0
                m_con, p_con = mask.copy(),pred.copy()
                m_con = cv2.drawContours(m_con, [m], -1, (255), 3)
                p_con = cv2.drawContours(p_con, [p], -1, (255), 3)
                for j in range(64):
                    for k in range(64):
                        m_pixel = m_con[j,k]
                        p_pixel = p_con[j,k]
                        if m_pixel == 255 and p_pixel == 255: TP+=1
                        if m_pixel == 255 and p_pixel == 0: FN+=1
                        if m_pixel == 0 and p_pixel == 255: FP+=1
                score = f_score(TP,FP,FN,f_score_beta)
                if score > f_score_tres:
                    TP_temp +=1
                    # match = np.argwhere(FP_temp==p)
                    # FP_temp = np.delete(FP_temp, match)
                    FP_temp = FP_temp[FP_temp!=p]
        FP_A += len(FP_temp)
        FN_A += (len(mask_contours)-TP_temp)
        TP_A += TP_temp
        if save_mode == 5: # save only contain true positive
            save = TP_temp > 0
        if save:
            if not os.path.exists('%s/'%save_dir):
                os.makedirs('%s/'%save_dir)

            fig = plt.figure()
            fig.suptitle('TP: %d, FN: %d, FP: %d'%(TP_temp, (len(mask_contours)-TP_temp), len(FP_temp)))
            # fig.suptitle('TP: %d, FN: %d, FP: %d'%(1,1,1))
            a = fig.add_subplot(1, 3, 1)
            gray = np.expand_dims(mask, axis=2)
            gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            imgplot = plt.imshow(gray)
            a.set_title('mask '+str(len(mask_contours)))

            a = fig.add_subplot(1, 3, 2)
            # gray = cv2.cvtColor(np.squeeze(pred),cv2.COLOR_GRAY2RGB)
            gray = np.expand_dims(pred, axis=2)
            gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            imgplot = plt.imshow(gray)
            a.set_title('Pred '+str(len(pred_contours)))

            a = fig.add_subplot(1, 3, 3)
            # gray = cv2.cvtColor(np.squeeze(img),cv2.COLOR_GRAY2RGB)
            imgplot = plt.imshow(img)
            a.set_title('image')

            new_name = img_name[:-4] + "__%02d" %(index)
            plt.savefig('%s/%s.png'%(save_dir, new_name))
            plt.close()

        index += 1
    # return 1,1,1
    return TP_A,FP_A,FN_A

def predict_folder(MODEL, base_path, save_mode=0, save_dir='./result'):
    files = glob.glob('%simages/*'%base_path)
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for path in tqdm(files, desc='images in folder', leave=False):
        img_name = os.path.basename(path)
        TP_A,FP_A,FN_A = predict(MODEL, base_path, img_name, save_mode, save_dir)
        # print(TP_A,FP_A,FN_A)
        TP_sum += TP_A
        FN_sum += FN_A
        FP_sum += FP_A
    return TP_sum, FP_sum, FN_sum
