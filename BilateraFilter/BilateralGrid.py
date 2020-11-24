import numpy as np
import cv2
import os
import time
import math
class BilateralGrid:
    def __init__(self,imageRgb):
        self.imageRgb = imageRgb
        # self.imageRgb = cv2.resize(self.imageRgb,dsize=(400,303))
        # self.imshow(imageRgb)
        self.rgb2grayCoeff = [0.0722,0.7152,0.2126]
        self.imageGray = self.rgb2Gray()
        self.imageRgbEnhanced = np.zeros_like(self.imageRgb)
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
        self.TMS = 2.5
    def rgb2Gray(self):
        imageGray = np.zeros(shape = (self.imageRgb.shape[0],self.imageRgb.shape[1]),dtype = np.int16)
        for i in range(self.imageRgb.shape[0]):
            for j in range(self.imageRgb.shape[1]):
                B = self.imageRgb[i][j][0]
                G = self.imageRgb[i][j][1]
                R = self.imageRgb[i][j][2]
                imageGray[i][j] = B * self.rgb2grayCoeff[0] + G * self.rgb2grayCoeff[1] + R * self.rgb2grayCoeff[2]
        return imageGray
    def clipBit(self,input,min,max):
        if input > max:
            output = max
            # print(input)
        elif input < min:
            output = min
        else:
            output = input
        return output



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
        gaussRangeKernel = [50.0,35.0]
        gaussSpaceKernel = [[50.0,35.0],
                            [35.0,15.0]]
        rKernel = 0.0
        sKernel = 0.0
        kernel = 0.0
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
                                            rKernel = gaussRangeKernel[abs(dz)]
                                            sKernel = gaussSpaceKernel[abs(dx)][abs(dy)]
                                            kernel = sKernel * rKernel
                                            gaussVal += self.vsum[i + dx][j + dy][k + dz] * kernel
                                            gaussWeight += self.pcnt[i + dx][j + dy][k + dz] * kernel
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
        rows = self.imageGray.shape[0]
        cols = self.imageGray.shape[1]
        idx_width = rows / self.GRID_X
        idy_width = cols / self.GRID_Y
        idx,idy,idz = 0.0,0.0,0.0
        idxLow,idxHigh = 0.0,0.0
        idyLow,idyHigh = 0.0,0.0
        idzLow,idzHigh = 0.0,0.0
        dx,dy,dz = 0.0,0.0,0.0
        """
        for row in range(rows):
            for col in range(cols):
                idx = int(row / idx_width)
                idy = int(col / idy_width)
                idz = self.imageGray[row][col] >> (self.BITWIDTH - self.GRIDZ_BITWIDTH)
                self.pcnt[idx][idy][idz] += 1
                self.vsum[idx][idy][idz] += lv[idz]
        """
        for row in range(rows):
            idx = row / idx_width
            idxLow = math.floor(idx)
            # print(row,idx,idxLow,math.ceil(idx))
            idxHigh = self.clipBit(math.ceil(idx),0,self.GRID_X - 1)
            xd = 0 if idxLow == idxHigh else ((idx - idxLow) / (idxHigh - idxLow))

            sum = 0
            for col in range(cols):
                idy = col / idy_width
                idyLow = math.floor(idy)
                idyHigh = self.clipBit(math.ceil(idy),0,self.GRID_Y - 1)

                yd = 0 if idyLow == idyHigh else ((idy - idyLow) / (idyHigh - idyLow))

                idz = self.imageGray[row][col] / (2 ** (self.BITWIDTH - self.GRIDZ_BITWIDTH))
                idzLow = math.floor(idz)
                idzHigh = self.clipBit(math.ceil(idz),0,self.GRID_Z - 1)
                # print("idylow,idyhigh =",idyLow,idyHigh)
                # print("idxlow,idxhigh =", idxLow, idxHigh)
                # print("idzlow,idzhigh =", idzLow, idzHigh)
                zd = 0 if idzLow == idzHigh else ((idz - idzLow) / (idzHigh - idzLow))
                sum = self.vgrid[idxLow][idyLow][idzLow] * (1-xd) * (1-yd) * (1-zd) + \
                      self.vgrid[idxLow][idyHigh][idzLow] * (1-xd) * (yd) * (1-zd) + \
                      self.vgrid[idxHigh][idyLow][idzLow] * (xd) * (1-yd) * (1-zd) + \
                      self.vgrid[idxHigh][idyHigh][idzLow] * (xd) * (yd) * (1-zd) + \
                      self.vgrid[idxLow][idyLow][idzHigh] * (1-xd) * (1-yd) * (zd) + \
                      self.vgrid[idxLow][idyHigh][idzHigh] * (1-xd) * (yd) * (zd) + \
                      self.vgrid[idxHigh][idyLow][idzHigh] * (xd) * (1-yd) *(zd) +\
                      self.vgrid[idxHigh][idyHigh][idzHigh] * (xd) * (yd) * (zd)
                # print("sum = ",sum)
                self.grayMask[row][col] = sum
        self.imshow(self.grayMask.astype(np.uint8))
    def ImageEnhance(self):
        grayEnhance = np.zeros(shape=(self.imageGray.shape[0],self.imageGray.shape[1]))
        for row in range(self.imageGray.shape[0]):
            for col in range(self.imageGray.shape[1]):
                grayEnhance[row][col] = self.imageGray[row][col] + self.TMS * (self.imageGray[row][col] - self.grayMask[row][col])
        #apply gain to rgb image
        for row in range(self.imageRgb.shape[0]):
            for col in range(self.imageRgb.shape[1]):
                for ch in range(self.imageRgb.shape[2]):
                    gain = grayEnhance[row][col] / self.imageGray[row][col]
                    tmp = self.imageRgb[row][col][ch] * gain
                    tmp = self.clipBit(tmp,0,1 << self.BITWIDTH)
                    self.imageRgbEnhanced[row][col][ch] = tmp
        self.imageRgbEnhanced = self.imageRgbEnhanced.astype(np.uint8)
        # self.imshow(self.imageRgbEnhanced)

        cv2.imshow("src",self.imageRgb)
        cv2.imshow("dst",self.imageRgbEnhanced)
        cv2.waitKey(0)





    def processImage(self):
        self.buildBilateralGrid()
        self.bilateralGridFilter()
        self.interpGrayImage()
        self.ImageEnhance()






if __name__ == '__main__':
    image = cv2.imread(r"E:\ImageFilter\Images\cloudy1.jpg")


    BilateralGrid(imageRgb=image).processImage()