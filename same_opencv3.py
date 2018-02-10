# encoding: utf-8

from panorama_opencv3 import Stitcher
import cv2
import os
import numpy as np

M_parg1=0.2
M_parg2=4

class Sort:
    @staticmethod
    def sorting(filePath):

        pathDir = os.listdir(filePath)
        fileLists = []
        fileLists_new = []
        for allDir in pathDir:
            file = os.path.join('%s%s' % (filePath, allDir))
            fileLists.append(file)
        print(fileLists, len(fileLists))

        def cut_left(image):
            image_left = image[0:image.shape[0], 0:image.shape[1] / 2]
            return image_left

        def cut_right(image):
            image_write = image[0:image.shape[0], image.shape[1] / 2:image.shape[1]]
            return image_write

        for index in np.arange(0, len(fileLists)):
            for i in np.arange(0, len(fileLists)):
                if i != index:
                    # 找第一张
                    imageA = cv2.imread(fileLists[index])
                    imageB = cv2.imread(fileLists[i])
                    imageA_left = cut_left(imageA)
                    imageB_right = cut_right(imageB)
                    stitcher = Stitcher()
                    (kpsA, featuresA) = stitcher.detectAndDescribe(imageA_left)
                    (kpsB, featuresB) = stitcher.detectAndDescribe(imageB_right)
                    M = stitcher.matchKeypoints_one(kpsA, kpsB, featuresA, featuresB, M_parg1,
                                                    M_parg2)
                    if M == True:
                        break
            if M == False:
                start = index
                fileLists_new.append(fileLists[start])
                print (fileLists_new)
                break

        # 排序
        for i in np.arange(0, len(fileLists) - 1):
            print i
            result = cv2.imread(fileLists_new[i])
            for index in np.arange(0, len(fileLists)):
                if index != start:
                    # if index != i:
                    imageA = result
                    imageB = cv2.imread(fileLists[index])
                    imageA_right = cut_right(imageA)
                    imageB_left = cut_left(imageB)
                    stitcher = Stitcher()
                    (kpsA, featuresA) = stitcher.detectAndDescribe(imageA_right)
                    (kpsB, featuresB) = stitcher.detectAndDescribe(imageB_left)
                    W = stitcher.matchKeypoints_one(kpsA, kpsB, featuresA, featuresB, M_parg1,
                                                    M_parg2)
                    if W == True:
                        fileLists_new.append(fileLists[index])
                        print (fileLists_new)
        print("fileLists_new:",fileLists_new)
        print("图片个数：",len(fileLists_new))
        return fileLists_new













'''
        #计算方差
        def getss(list):
            #计算平均值
            avg=sum(list)/len(list)
            #定义方差变量ss，初值为0
            ss=0
            #计算方差
            for l in list:
                ss+=(l-avg)*(l-avg)/len(list)   
            #返回方差
            return ss

        #获取每行像素平均值  
        def getdiff(img):
            #定义边长
            Sidelength=30
            #缩放图像
            img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)
            #灰度处理
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #avglist列表保存每行像素平均值
            avglist=[]
            #计算每行均值，保存到avglist列表
            for i in range(Sidelength):
                avg=sum(gray[i])/len(gray[i])
                avglist.append(avg)
            #返回avglist平均值   
            return avglist


        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''
