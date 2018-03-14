# encoding: utf-8

from panorama_opencv3 import Stitcher
import cv2
import os
import numpy as np
from same_opencv3 import Sort
import time

S_parg1=0.6
S_parg2=8

start = time.time()
filePath = '../Image_Stitching/pic_8/'
fileLists_new = Sort.sorting(filePath)
result = cv2.imread(fileLists_new[0])

for index in np.arange(0, len(fileLists_new)):
    if index > 0:
        imageA = result
        imageB = cv2.imread(fileLists_new[index])
        stitcher = Stitcher()
        (result, vis) = stitcher.stitch([imageA, imageB], S_parg1, S_parg2, showMatches=True)
        cv2.imshow("keypoint Matches", vis)
        cv2.imshow("Result", result)
        cv2.waitKey(0)

        if index == len(fileLists_new) - 1:
            end = time.time()
            print ("时间：")
            print end - start
            cv2.imshow("keypoint Matches", vis)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
