import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import math

BASE = './result'

if not os.path.exists('%s/whole_body/'%BASE):
    os.makedirs('%s/whole_body/'%BASE)

FNAME = '1.0014'

height = 896
width = 512
h,w = 64,64
hloop = int(math.ceil(width/w)) #Horizontal loop
vloop = int(math.ceil(height/h)) #Vertical loop

for k in range(32):
    result = np.array([[[0 for x in range(3)] for y in range(width)]for z in range(height)])
    for i in range(hloop):
        for j in range(vloop):
            img_path = BASE + '/%s_%s_%s__%s.png'%(FNAME,i,j,14+k)
            img = np.array(cv2.imread(img_path, 1))
            result[h*j:h*(j+1), w*i:w*(i+1):] = img
    new_name = '/1.00%s'%(14+k)
    cv2.imwrite('./result/whole_body/%s.png'%new_name, result)