import os
import cv2
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm

## global parameters
f_score_tres = 0.6
f_score_beta = 2

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

    TP_A, FP_A, FN_A = 0, 0, 0 # TP,FP,FN of 32 images (z-axis)
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
        p_match, m_match = [],[]

        for m in range(len(mask_contours)):
            for p in range(len(pred_contours)):
                TP,FP,FN = 0,0,0 # TP,FP,FN of current contour
                m_con, p_con  = np.zeros((mask.shape[0],mask.shape[1])), np.zeros((pred.shape[0],pred.shape[1]))
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
            fig, a = plt.subplots(figsize=(10, 4))
            a.axis('off')
            fig.suptitle('%s => TP: %d, FN: %d, FP: %d'%(new_name,TP_temp, FN_temp, FP_temp), fontsize = 16)

            a = fig.add_subplot(1, 3, 1)
            gray = np.expand_dims(mask, axis=2)
            gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            imgplot = plt.imshow(gray)
            a.set_title('Ground truth', loc='center')
            a.axes.xaxis.set_visible(False)
            a.axes.yaxis.set_visible(False)

            a = fig.add_subplot(1, 3, 2)
            gray = np.expand_dims(pred, axis=2)
            gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            imgplot = plt.imshow(gray)
            a.set_title('Prediction', loc='center')
            a.axes.xaxis.set_visible(False)
            a.axes.yaxis.set_visible(False)

            a = fig.add_subplot(1, 3, 3)
            imgplot = plt.imshow(img)
            a.set_title('Image', loc='center')
            a.axes.xaxis.set_visible(False)
            a.axes.yaxis.set_visible(False)

            plt.subplots_adjust(left=0.01, bottom=0, right=0.99, top=0.9, wspace=0.1, hspace=0.1)
            plt.savefig('%s/%s.png'%(save_dir, new_name))
            plt.close()

        index += 1
    return TP_A,FP_A,FN_A

def predict_folder(MODEL, base_path, save_mode=0, save_dir='./result'):
    files = glob.glob('%simages/*'%base_path)
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for path in tqdm(files, desc='images in folder', leave=False):
        img_name = os.path.basename(path)
        TP_A,FP_A,FN_A = predict(MODEL, base_path, img_name, save_mode, save_dir)
        TP_sum += TP_A
        FN_sum += FN_A
        FP_sum += FP_A
    return TP_sum, FP_sum, FN_sum
