#Implemention for cv2 bilateralFiter
import cv2
import os
import numpy as np
import math
class BilateralFilter():
    def __init__(self,Image,ImageBIT,sigmaColor,sigmaSpace):
        self.Image = Image
        self.maxVal = 1 << 8
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.deltaGaussColorTab = {}
        self.deltaGaussDistaTab = [][]
    def clip_bits(self,input):
        if input > self.maxVal:
            return maxVal
        elif input < 0:
            return 0
        else:
            return input
    def lookUpGaussTabColor(self):
        deltaGauss = {}
        for delta in range(self.maxVal):
            wt = math.exp(-delta ** 2 / (2 * (self.sigmaColor ** 2)))
            deltaGauss[delta] = wt
        return deltaGaussColorTab

    def lookUpGassTabDistance(self):
        pass






    def BilateralFilter(self,Image):
        # input parms
        windowLenth = 7
        sigmaColor = 25
        sigmaSpace = 9
        maskImageMatrix = np.zeros(shape=(Image.shape[0],Image.shape[1]),dtype=np.uint32)
        #prcess image

if __name__ == '__main__':
    path = "./Images"
    image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imageFilter = BilateralFilter(imageGray)