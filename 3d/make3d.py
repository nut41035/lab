import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import random as r

BASE = './data/processed/new/'
# fpath = glob.glob('%simages/*' %BASE)

if not os.path.exists('%s/train/images/'%BASE):
    os.makedirs('%s/train/images/'%BASE)
if not os.path.exists('%s/train/masks/'%BASE):
    os.makedirs('%s/train/masks/'%BASE)
if not os.path.exists('%s/test/images/'%BASE):
    os.makedirs('%s/test/images/'%BASE)
if not os.path.exists('%s/test/masks/'%BASE):
    os.makedirs('%s/test/masks/'%BASE)
if not os.path.exists('%s/validation/images/'%BASE):
    os.makedirs('%s/validation/images/'%BASE)
if not os.path.exists('%s/validation/masks/'%BASE):
    os.makedirs('%s/validation/masks/'%BASE)

fname = []
for i in range(8):
    for j in range(14):
        fname.append('1.0001_%s_%s' %(i,j))
        fname.append('1.0014_%s_%s' %(i,j))
        fname.append('1.0029_%s_%s' %(i,j))
        fname.append('2.0001_%s_%s' %(i,j))
        fname.append('2.0009_%s_%s' %(i,j))
        fname.append('2.0019_%s_%s' %(i,j))

r.shuffle(fname)
l = len(fname)
train = fname[:int(l*0.7)]
validation = fname[int(l*0.7):int(l*0.9)]
test = fname[int(l*0.9):]

def make3d_img(name,save_folder):
    n = int(name[4:6])
    img_path = BASE + 'images/' + name + '.jpg'
    img = cv2.imread(img_path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = np.array(gray)
    res = np.expand_dims(res, axis=2)
    for i in range(1,32):
        tmp = name[:4] + "%02d" %(n+i) + name[6:]
        img_path = BASE + 'images/' + tmp + '.jpg'
        img = cv2.imread(img_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(gray)
        img = np.expand_dims(img, axis=2)
        res = np.concatenate((img,res), 2, None)
    res = np.expand_dims(res, axis=3)
    new_name = name + '.npy'
    np.save('%s/%s/images/%s' %(BASE,save_folder,new_name),res)
    return res

def make3d_mask(name,save_folder):
    n = int(name[4:6])
    mask_path = BASE + 'masks/' + name + '.npy'
    res = np.load(mask_path)
    res = np.expand_dims(res, axis=2)
    for i in range(1,32):
        tmp = name[:4] + "%02d" %(n+i) + name[6:]
        mask_path = BASE + 'masks/' + tmp + '.npy'
        img = np.load(mask_path)
        img = np.expand_dims(img, axis=2)
        res = np.concatenate((img,res), 2, None)             
    res = np.expand_dims(res, axis=3)
    new_name = name + '.npy'
    np.save('%s/%s/masks/%s' %(BASE,save_folder,new_name),res)
    return res

for name in tqdm(train, desc='slicing img', leave=False):
    res = make3d_img(name,'train')
    res2 = make3d_mask(name,'train')
print('training image: %d images'%len(train))
print('             - image dim: %s'%str(res.shape))
print('             - mask  dim: %s'%str(res2.shape))
for name in tqdm(validation, desc='slicing img', leave=False):
    res = make3d_img(name,'validation')
    res2 = make3d_mask(name,'validation')
print('validation image: %d images'%len(validation))
print('             - image dim: %s'%str(res.shape))
print('             - mask  dim: %s'%str(res2.shape))
for name in tqdm(test, desc='slicing img', leave=False):
    res = make3d_img(name,'test')
    res2 = make3d_mask(name,'test')
print('testing image: %d images'%len(test))
print('             - image dim: %s'%str(res.shape))
print('             - mask  dim: %s'%str(res2.shape))