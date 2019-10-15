import cv2 as cv
import argparse
import glob
import os
import numpy as np

parser = argparse.ArgumentParser(description='''
    Run this script with a folder containing only .jpg file
    to convert it to binary format. output is saved at ./data/GT
    ''')

parser.add_argument('-t', type=int, default=127,
    help='treshold value')
parser.add_argument('fName', help='path to folder from ./')

args = parser.parse_args()
t = args.t
fName = args.fName
files = glob.glob('./%s/*' %fName)
print(files)

if not os.path.exists('./data/GT'):
    os.makedirs('./data/GT')
if not os.path.exists('./data/np'):
    os.makedirs('./data/np')

for imgName in files:
    src = cv.imread(imgName,1)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret,thresh1 = cv.threshold(src,t,255,cv.THRESH_BINARY)
    name = './data/GT/%s.png' %os.path.basename(imgName)[:-4]
    npName = './data/np/%s' %os.path.basename(imgName)[:-4]
    np.save(npName, thresh1)
    cv.imwrite(name, thresh1)
    print(os.path.basename(name))
