import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

## global parameters
f_score_tres = 0.6
f_score_beta = 2

def predict(MODEL, base_path, img_name, save_mode=0, save_dir='./result'):
    """ save_mode : 0 = not save
                    1 = save all
                    2 = save only cancer in mask
                    3 = save only predicted something
                    4 = save only contain cancer || prediced something
                    5 = save only contain true positive
    """
    # load image and mask
    image_path = os.path.join(base_path, 'images/', img_name)
    image = np.load(image_path)
    image = np.float32(image)
    image = np.expand_dims(image, axis=0)

    mask_path = os.path.join(base_path, 'masks/', img_name)
    masks = np.load(mask_path)
    masks = np.where(masks < 0.7, 0, 255)
    masks = np.squeeze(masks)

    # prediction
    prediction = MODEL.predict(image)
    prediction = np.squeeze(prediction)
    prediction = np.where(prediction < 0.7, 0, 255)

    index = int(img_name[2:6]) # slice depth

    TP_A, FP_A, FN_A, TP_P, FP_P, FN_P = 0, 0, 0, 0, 0, 0 # TP,FP,FN of 32 images (z-axis) _A = area, _P = pixel
    for i in range(32):
        img = image[0,:,:,i,:].astype(np.uint8)

        mask = masks[:,:,i].astype(np.uint8)
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        pred = prediction[:,:,i].astype(np.uint8)
        pred_contours, _ = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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

        TP_temp = 0 # TP of current slice
        TP, FP, FN, TP_temp, p_match, m_match, match = match_contour(mask_contours, pred_contours, mask)
        TP_P += TP
        FP_P += FP
        FN_P += FN
        FP_temp = (len(pred_contours)-len(list(dict.fromkeys(p_match)))) # list(dict.fromkeys(array)) remove duplicate
        FN_temp = (len(mask_contours)-len(list(dict.fromkeys(m_match))))
        TP_A += TP_temp
        FP_A += FP_temp
        FN_A += FN_temp
        if save_mode == 5: # save only contain true positive
            save = TP_temp > 0
        if save:
            if not os.path.exists('%s/'%save_dir):
                os.makedirs('%s/'%save_dir)

            new_name = img_name[:-4] + "__%02d" %(index)
            save_grid(save_dir, new_name, TP_temp, FN_temp, FP_temp, img, mask, pred, match)

        index += 1
    return TP_A,FP_A,FN_A,TP_P,FP_P,FN_P

def predict_folder(MODEL, base_path, save_mode=0, save_dir='./result'):
    files = glob.glob('%simages/*'%base_path)
    TP_A_sum, FP_A_sum, FN_A_sum = 0,0,0
    TP_P_sum, FP_P_sum, FN_P_sum = 0,0,0
    for path in tqdm(files, desc='images in folder', leave=False):
        img_name = os.path.basename(path)
        TP_A, FP_A, FN_A, TP_P, FP_P, FN_P = predict(MODEL, base_path, img_name, save_mode, save_dir)
        TP_A_sum += TP_A
        FN_A_sum += FN_A
        FP_A_sum += FP_A
        TP_P_sum += TP_P
        FN_P_sum += FN_P
        FP_P_sum += FP_P
    return TP_A_sum, FP_A_sum, FN_A_sum, TP_P_sum, FP_P_sum, FN_P_sum

def match_contour(mask_contours, pred_contours, mask):
    p_match, m_match = [],[]
    match = []
    TP_temp = 0
    TP_P, FP_P, FN_P = 0, 0, 0
    for m in range(len(mask_contours)):
        for p in range(len(pred_contours)):
            TP,FP,FN = 0,0,0 # TP,FP,FN of current contour
            m_con, p_con  = np.zeros_like(mask), np.zeros_like(mask)
            m_con = cv2.drawContours(m_con, mask_contours, m, (255), -1)
            p_con = cv2.drawContours(p_con, pred_contours, p, (255), -1)
            for j in range(64):
                for k in range(64):
                    m_pixel = m_con[j,k]
                    p_pixel = p_con[j,k]
                    if m_pixel == 255 and p_pixel == 255: TP+=1
                    if m_pixel == 255 and p_pixel == 0: FN+=1
                    if m_pixel == 0 and p_pixel == 255: FP+=1
            score = f_score(TP,FP,FN,f_score_beta) # f(beta) score
            if score > f_score_tres:
                TP_temp +=1
                m_match.append(m)
                p_match.append(p)
                match.append(mask_contours[m])
            TP_P += TP
            FP_P += FP
            FN_P += FN
    return TP_P, FP_P, FN_P, TP_temp, p_match, m_match, match

def save_grid(save_dir, name, TP, FN, FP, img, mask, pred, match):
    fig, a = plt.subplots(figsize=(8, 9))
    a.axis('off')
    fig.suptitle('%s => TP: %d, FN: %d, FP: %d'%(name,TP, FN, FP), fontsize = 16)

    a = fig.add_subplot(2, 2, 1)
    gray = np.expand_dims(mask, axis=2)
    gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    gray = mask_match(gray, match)
    imgplot = plt.imshow(gray)
    a.set_title('Ground truth', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    a = fig.add_subplot(2, 2, 2)
    gray = np.expand_dims(pred, axis=2)
    gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    imgplot = plt.imshow(gray)
    a.set_title('Prediction', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    a = fig.add_subplot(2, 2, 3)
    combined = combine(pred, mask)
    imgplot = plt.imshow(combined)
    a.set_title('combine', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    a = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(img)
    a.set_title('Image', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    plt.subplots_adjust(left=0.01, bottom=0, right=0.99, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig('%s/%s.png'%(save_dir, name))
    plt.close()

def mask_match(mask, match):
    for i in match:
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),0)
    return mask

def combine(predicted, mask):
    combined = np.zeros((64,64,3))
    TP = [51,153,51] # green
    FN = [255,77,77] # red
    FP = [255,255,128] # yellow
    for i in range(64):
        for j in range(64):
            if predicted[i][j] == 255 and mask[i][j] == 255: combined[i][j] = TP
            if predicted[i][j] == 0 and mask[i][j] == 255: combined[i][j] = FN
            if predicted[i][j] == 255 and mask[i][j] == 0: combined[i][j] = FP

    combined = np.int32(combined)
    return combined

def f_score(tp,fp,fn,b):
    #recall is b times as important as precision
    try:
        p = tp/(tp+fp)
    except ZeroDivisionError:
        p = 0

    try:
        r = tp/(tp+fn)
    except ZeroDivisionError:
        r = 0
    
    try:
        x = (1+b*b)*((p*r)/(b*b*p+r))
    except ZeroDivisionError:
        x = 0
    return x