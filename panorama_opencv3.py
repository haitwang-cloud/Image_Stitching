# encoding: utf-8

import numpy as np
import cv2
import math


class Stitcher:
    distance = 0

    def stitch(self, images, ratio, reprojThresh, showMatches=False):
        # 打开图像，检测关键点并提取本地不变描述符
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 匹配两图图像中的特征
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果没有足够的匹配点
        if M is None:
            return None

        # 用一个透视矩阵来拼接图像
        (matches, H, status) = M
        dist = self.distance(imageA, imageB, kpsA, kpsB, matches, status)
        print '绿线长度：'
        print dist

        #        t=dist-imageA.shape[1]
        #        print '重合部分长度：'
        #        print t
        result = np.zeros((imageA.shape[0], dist, 3), imageA.dtype)
        while (dist<imageB.shape[1]):
            dist+=1
        result = cv2.warpPerspective(imageA, H,
                                     (dist, imageA.shape[0]))

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        print "透视矩阵："
        print H
        # print result.shape
        # print '----------'
        # 检查匹配点是否应可视化
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # 将全景图和可视化的元组返回给调用函数
            return (result, vis)

        return result

    def detectAndDescribe(self, image):

        # 我们使用了DoG关键点检测和SIFT特征提取。 `

        # 将图像转化为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # create SIFT and detect/compute
        surf = cv2.xfeatures2d.SURF_create()
        kps, features = surf.detectAndCompute(gray, None)

        # 将关键点KeyPoints从KeyPoint对象转换为NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        return (kps, features)

    # 匹配
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, raprojThresh):

        # FLANN matcher parameters
        # FLANN_INDEX_KDTREE = 0
        indexParams = dict(algorithm=0, trees=5)
        searchParams = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        rawMatches = flann.knnMatch(featuresA, featuresB, k=2)


        # 计算原始匹配并初始化实际匹配的列表
        # opencv构造了特性匹配器DescriptorMatcher_create。
        # BruteForce值表示,计算两个图像中所有特征向量之间的欧氏距离，
        # 并找到最小距离的描述符对(为每个特征点找到前两个距离最小的特征描述符)
        # matcher = cv2.DescriptorMatcher_create("BruteForce")
        # rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # print "rawMatches:"
        # print rawMatches

        matches = []

        # intqueryIdx;  //此匹配对应的查询图像的特征描述子索引(对应特征点的下标)
        # inttrainIdx;   //此匹配对应的训练(模板)图像的特征描述子索引
        # 确保距离在一定比例内(即Lowe's ratio test)
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))  # trainIdx是featuresB的索引，queryIdx是featuresA的索引
        #                print "matches"
        #                print matches
        #                print len(matches)
        #                print "B图特征点序号"
        #                print m[0].trainIdx
        #                print "A图特征点序号"
        #                print m[0].queryIdx

        print("匹配点数量：")
        print(len(matches))

        # 计算匹配点的homography
        if len(matches) > 4:
            # 构建两组点(描述子不但包含关键点，也包括关键点周围对其有贡献的邻域点。)
            ptsA = np.float32([kpsA[i] for (_, i) in matches])

            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算单映性矩阵（findHomography函数参数包括源，目标，筛选方法，容错阈值(超过该阈值就认为是 outlier)）
            # 返回矩阵和每对匹配点的状态
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, raprojThresh)

            return (matches, H, status)
        return None

    # 将对应关键点可视化
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化输出可视化图像
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop the matches（status是每个匹配点的状态）
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                # 画出匹配
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

    def distance(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        dis = []
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            dist = ptB[0] - ptA[0]
            dis.append(dist)

        dist_aver = sum(dis) / len(matches)

        return dist_aver

    def matchKeypoints_one(self, kpsA, kpsB, featuresA, featuresB, ratio,num):
        # 计算原始匹配并初始化实际匹配的列表
        # opencv构造了特性匹配器DescriptorMatcher_create。
        # BruteForce值表示,计算两个图像中所有特征向量之间的欧氏距离，
        # 并找到最小距离的描述符对(为每个特征点找到前两个距离最小的特征描述符)
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        # print "rawMatches:"
        # print rawMatches
        matches = []

        # intqueryIdx;  //此匹配对应的查询图像的特征描述子索引(对应特征点的下标)
        # inttrainIdx;   //此匹配对应的训练(模板)图像的特征描述子索引
        # 确保距离在一定比例内(即Lowe's ratio test)
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                # trainIdx是featuresB的索引，queryIdx是featuresA的索引
                # 计算匹配点的homography
        if len(matches) > num:
            # 构建两组点(描述子不但包含关键点，也包括关键点周围对其有贡献的邻域点。)
            # 计算单映性矩阵（findHomography函数参数包括源，目标，筛选方法，容错阈值(超过该阈值就认为是 outlier)）
            # 返回矩阵和每对匹配点的状态
            return True
        return False
