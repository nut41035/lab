import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import random as r

BASE = './data/processed/'
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
test_fname = []
for i in range(8):
    for j in range(14):
        fname.append('1.0001_%s_%s' %(i,j))
        fname.append('1.0029_%s_%s' %(i,j))
        fname.append('2.0001_%s_%s' %(i,j))
        fname.append('2.0019_%s_%s' %(i,j))
        fname.append('2.0009_%s_%s' %(i,j))
        fname.append('1.0014_%s_%s' %(i,j))
r.shuffle(fname)
l = len(fname)
train = fname[:int(l*0.7)]
validation = fname[int(l*0.7):int(l*0.9)]
test = fname[int(l*0.9):]

def make3d_img(name,save_folder):
    n = int(name[4:6])
    img_path = BASE + 'images/' + name + '.jpg'
    res = np.array(cv2.imread(img_path, 1))
    res = np.expand_dims(res, axis=3)
    for i in range(1,32):
        tmp = name[:4] + "%02d" %(n+i) + name[6:]
        img_path = BASE + 'images/' + tmp + '.jpg'
        img = np.array(cv2.imread(img_path, 1))
        img = np.expand_dims(img, axis=3)
        res = np.concatenate((img,res), 3, None)
    res = res.swapaxes(2,3)
    new_name = name + '.npy'
    np.save('%s/%s/images/%s' %(BASE,save_folder,new_name),res)

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
    new_name = name + '.npy'
    np.save('%s/%s/masks/%s' %(BASE,save_folder,new_name),res)

for name in tqdm(train, desc='slicing img', leave=False):
    make3d_img(name,'train')
    make3d_mask(name,'train')
for name in tqdm(validation, desc='slicing img', leave=False):
    make3d_img(name,'validation')
    make3d_mask(name,'validation')
for name in tqdm(test, desc='slicing img', leave=False):
    make3d_img(name,'test')
    make3d_mask(name,'test')