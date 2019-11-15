
# img = cv2.imread("lenna.png")
# crop_img = img[y:y+h, x:x+w]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
import cv2 as cv
import argparse
import glob
import os
import math
import numpy as np
import itertools


parser = argparse.ArgumentParser(description='''
    Run this script with a folder containing only .jpg file
    to slice it into a smaller images PATCHES. 
    data structure should be like this'''
    )

parser.add_argument('-width', type=int, default=64,
    help='Width of croped image')
parser.add_argument('-height', type=int, default=64,
    help='Hight of croped image')
parser.add_argument('-t', type=bool, default=False,
    help='Crop only images that contain cancer if True')
parser.add_argument('fName', help='path to input folder from ./')

# setup parameters
args = parser.parse_args()
w = args.width
h = args.height
t = args.t
fName = args.fName
files = glob.glob('./%simages/*' %fName)
files_GT = glob.glob('./%smasks/*' %fName)
files_np = glob.glob('./%snp/*' %fName)
print(files_np)
# how many times it should loop
hloop = int(math.floor(480/w)) #Horizontal loop
vloop = int(math.floor(864/h)) #Vertical loop

if not os.path.exists('./%s/sliced/images'%fName):
    os.makedirs('./%s/sliced/images'%fName)
if not os.path.exists('./%s/sliced/masks'%fName):
    os.makedirs('./%s/sliced/masks'%fName)
if not os.path.exists('./%s/sliced/np'%fName):
    os.makedirs('./%s/sliced/np'%fName)

if (t):
    # # GT images
    for path in files_GT:
        imgName = os.path.basename(path)[0:-4]
        imgPath = os.path.dirname(path)[0:-6] + '/images/'
        img = cv.imread(imgPath+imgName+'.jpg',1)
        gt = cv.imread(path,1)
        for i in range(hloop):
            for j in range(vloop):
                crop_gt = gt[h*j:h*(j+1), w*i:w*(i+1)]
                flatten = crop_gt.flatten()
                if max(flatten) > 100:
                    crop_img = img[h*j:h*(j+1), w*i:w*(i+1)]
                    newName = imgName + '_%d_%d.jpg'%(i,j)
                    newNameGT = imgName + '_%d_%d.png'%(i,j)
                    cv.imwrite('./%s/sliced/images/%s' %(fName,newName), crop_img)
                    cv.imwrite('./%s/sliced/masks/%s' %(fName,newNameGT), crop_gt)
else:
    #image files
    for imgName in files:
        img = cv.imread(imgName,1)
        for i in range(hloop):
            for j in range(vloop):
                crop_img = img[h*j:h*(j+1), w*i:w*(i+1)]
                newName = os.path.basename(imgName)[:-4] + '_%d_%d'%(i,j)
                cv.imwrite('./%s/sliced/images/%s.jpg' %(fName,newName), crop_img)

    #GT files
    for imgName in files_GT:
        img = cv.imread(imgName,1)
        for i in range(hloop):
            for j in range(vloop):
                crop_img = img[h*j:h*(j+1), w*i:w*(i+1)]
                newName = os.path.basename(imgName)[:-4] + '_%d_%d'%(i,j)
                cv.imwrite('./%s/sliced/masks/%s.png' %(fName,newName), crop_img)


    # NP files
    for imgName in files_np:
        img = np.load(imgName)
        for i in range(hloop):
            for j in range(vloop):
                crop_img = img[h*j:h*(j+1), w*i:w*(i+1)]
                newName = os.path.basename(imgName)[:-4] + '_%d_%d'%(i,j)
                np.save('./%s/sliced/np/%s' %(fName,newName), crop_img)
