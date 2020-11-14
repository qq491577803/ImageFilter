#Implemention for cv2 bilateralFiter
import cv2
import numpy as np
import time
class BilateralFilter():
    def __init__(self,Image,ImageBit,sigmaColor,sigmaSpace,radius):
        self.Image = Image
        self.ImageDeep = 1 << ImageBit
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.radius = radius
        self.chanle = 3 if len(self.Image.shape) == 3 else 1
    def buildGaussSapce(self,sigmaSpce,radius):
        gaussSpace = np.zeros(shape=(2 * radius + 1,2 * radius + 1),dtype=np.float32)
        leftIndex = - radius
        rightIndex =  radius
        for i in range(leftIndex,rightIndex + 1):
            for j in range(leftIndex,rightIndex + 1):
                gaussSpace[i - leftIndex][j - leftIndex] = np.exp(-(i ** 2 + j ** 2) / (2 * sigmaSpce ** 2))
        return gaussSpace

    def buildGaussColor(self,sigmaColor):
        gaussColor = []
        for delta in range(self.ImageDeep + 1):
            gaussColor.append(np.exp(- delta ** 2 / (2 * sigmaColor ** 2)).astype(np.float32))
        return gaussColor
    def clipBit(self,input):

        input = self.ImageDeep if input > self.ImageDeep else input
        input = 0 if input < 0 else input
        return int(input)
    def filter(self):
        gaussSapce = self.buildGaussSapce(self.sigmaSpace,self.radius)
        gaussColor = self.buildGaussColor(self.sigmaColor)
        margin = self.radius
        leftIndex = - self.radius
        rightIndex = self.radius
        dstImage = self.Image.copy()
        for row in range(margin,self.Image.shape[0] - margin):
            for col in range(margin,self.Image.shape[1] - margin):
                if self.chanle == 1:
                    centerVal = self.Image[row][col]
                    imageVal = 0
                    wtSum = 0
                    for i in range(leftIndex,rightIndex + 1):
                        for j in range(leftIndex,rightIndex + 1):
                            tmpVal = self.Image[row + i][col + j]
                            colorWt = gaussColor[int(abs(tmpVal - centerVal))]
                            spaceWt = gaussSapce[i][j]
                            wt = colorWt * spaceWt
                            imageVal = imageVal + tmpVal * wt
                            wtSum = wtSum + wt
                    imageVal = imageVal / wtSum
                    dstImage[row][col] = self.clipBit(imageVal)
                else:
                    centerB,centerG,centerR = self.Image[row][col][:]
                    imageValB,imageValG,imageValR = 0,0,0
                    wtSumB,wtSumG,wtSumR = 0,0,0
                    for i in range(leftIndex,rightIndex + 1):
                        for j in range(leftIndex,rightIndex + 1):
                            tmpValB,tmpValG,tmpValR = self.Image[row + i][col + j][:]
                            colorWtB,colorWtG,colorWtR = gaussColor[int(abs(tmpValB - centerB))],\
                                                         gaussColor[int(abs(tmpValG - centerG))],\
                                                         gaussColor[int(abs(tmpValR - centerR))]
                            spaceWt = gaussSapce[i][j]
                            wtB,wtG,wtR = colorWtB * spaceWt,colorWtG * spaceWt,colorWtR * spaceWt
                            imageValB,imageValG,imageValR = imageValB + tmpValB * wtB,\
                                                            imageValG + tmpValG * wtG,\
                                                            imageValR + tmpValR * wtR
                            wtSumB, wtSumG,wtSumR= wtSumB + wtB,wtSumG + wtG,wtSumR + wtR
                    imageValB,imageValG,imageValR = imageValB / wtSumB,imageValG / wtSumG,imageValR / wtSumR
                    dstImage[row][col][:] = self.clipBit(imageValB),self.clipBit(imageValG),self.clipBit(imageValR)
        return dstImage
if __name__ == '__main__':
    path = "../Images/face.jpg"
    imageGray = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    startT = time.time()
    imageFilter = BilateralFilter(imageGray,ImageBit = 8,sigmaColor = 25,sigmaSpace = 9,radius = 3).filter()
    print("Total cost time :",time.time() - startT)
    cv2.imshow("srcImage",imageGray)
    cv2.imshow("dstImage",imageFilter)
    cv2.waitKey(0)