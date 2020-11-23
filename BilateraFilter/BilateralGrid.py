
import numpy as np
import cv2
import os
import time

class BilateralGrid:
    def __init__(self,imageRgb):
        self.imageRgb = imageRgb
        self.imageRgb = cv2.resize(self.imageRgb,dsize=(400,303))
        # self.imshow(imageRgb)
        self.rgb2grayCoeff = [0.0722,0.7152,0.2126]
        self.imageGray = self.rgb2Gray()
        print("imageGray.shape:",self.imageGray.shape[0],self.imageGray.shape[1])
        self.GRID_X = 12
        self.GRID_Y = 8
        self.GRID_Z = 9
        self.pcnt = np.zeros(shape=(self.GRID_X,self.GRID_Y,self.GRID_Z),dtype = np.int64)
        self.vsum = np.zeros(shape=(self.GRID_X,self.GRID_Y,self.GRID_Z),dtype = np.int64)
        self.vgrid= np.zeros(shape=(self.GRID_X,self.GRID_Y,self.GRID_Z),dtype = np.float32)
        self.BITWIDTH = 8
        self.GRIDZ_BITWIDTH = 3
        self.debug = True
        self.initCnt = 1
        self.grayMask = np.zeros(shape=(self.imageGray.shape[0],self.imageGray.shape[1]))
    def rgb2Gray(self):
        imageGray = np.zeros(shape = (self.imageRgb.shape[0],self.imageRgb.shape[1]),dtype = np.int16)
        for i in range(self.imageRgb.shape[0]):
            for j in range(self.imageRgb.shape[1]):
                B = self.imageRgb[i][j][0]
                G = self.imageRgb[i][j][1]
                R = self.imageRgb[i][j][2]
                imageGray[i][j] = B * self.rgb2grayCoeff[0] + G * self.rgb2grayCoeff[1] + R * self.rgb2grayCoeff[2]
        return imageGray
    def imshow(self,image):
        cv2.imshow("image",image.astype(np.uint8))
        cv2.waitKey(0)
    def buildBilateralGrid(self):
        rows = self.imageGray.shape[0]
        cols = self.imageGray.shape[1]
        idx_width = rows / self.GRID_X
        idy_width = cols / self.GRID_Y
        print("idx_width,idy_width:",idx_width,idy_width)
        lv = [int(i * ((1 << self.BITWIDTH) / (self.GRID_Z - 1))) for i in range(self.GRID_Z)]
        print("lv:",lv)
        #init 3D Vgrid
        for idx in range(self.GRID_X):
            for idy in range(self.GRID_Y):
                for idz in range(self.GRID_Z):
                    self.pcnt[idx][idy][idz] += self.initCnt
                    self.vsum[idx][idy][idz] += self.initCnt * lv[idz]
        #build 3D Vgrid via grayImage
        for row in range(rows):
            for col in range(cols):
                idx = int(row / idx_width)
                idy = int(col / idy_width)
                idz = self.imageGray[row][col] >> (self.BITWIDTH - self.GRIDZ_BITWIDTH)
                self.pcnt[idx][idy][idz] += 1
                self.vsum[idx][idy][idz] += lv[idz]
        if (self.debug):
            print("pcnt:")
            for row in range(self.GRID_X):
                for col in range(self.GRID_Y):
                    for idz in range(self.GRID_Z):
                        print(self.pcnt[row][col][idz],end = ",")
            print()
            print("vsum:")
            for row in range(self.GRID_X):
                for col in range(self.GRID_Y):
                    for idz in range(self.GRID_Z):
                        print(self.vsum[row][col][idz],end=",")
            print()
        print("Successfully buid 3D vgrid.")
    def bilateralGridFilter(self):
        print("Start to 3D grid filter.")
        gaussRangeKernel = []
        gaussSpaceKernel = []
        for i in range(self.GRID_X):
            for j in range(self.GRID_Y):
                for k in range(self.GRID_Z):
                    gaussVal = 0.0
                    gaussWeight = 0.0
                    for dx in range(-1,2):
                        if not((i==0 and dx == -1) or (i == self.GRID_X -1 and dx == 1)):
                            for dy in range(-1,2):
                                if not((j == 0 and dy == -1) or (j == self.GRID_Y - 1 and dy == 1)):
                                    for dz in range(-1,2):
                                        if not((k == 0 and dz == -1) or (k == self.GRID_Z - 1 and dz == 1)):
                                            kernel = x * dy
                                            gaussVal +=
                                            gaussWeight +=
                    self.vgrid[i][j][k] = gaussVal / gaussWeight
        if(self.debug):
            print("vgrid:")
            for idx in range(self.GRID_X):
                for idy in range(self.GRID_Y):
                    for idz in range(self.GRID_Z):
                        print(self.vgrid[idx][idy][idz],end = ",")
            print()

        print("Sucessfully vgrid filter.")
    def interpGrayImage(self):
        #interp gray image to buid gray mask

        pass



    def grayImageEhance(self):
        pass
    def processImage(self):
        self.buildBilateralGrid()
        self.interpGrayImage()
        self.GrayImageEnhance()






if __name__ == '__main__':
    image = cv2.imread(r"E:\ImageFilter\Images\cloudy1.jpg")


    BilateralGrid(imageRgb=image).processImage()