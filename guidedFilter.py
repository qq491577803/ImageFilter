import numpy as np
import cv2
import matplotlib.pyplot as plt


class GuidedFilter:
    def __init__(self,I,p,radius,eps):
        self.I = np.array(I ,dtype=np.float32)
        self.p = np.array(p ,dtype=np.float32)
        self.eps = eps
        self.radius = radius

    def rgb2Gray(self,imageRgb):
        imageGray = np.zeros(shape = (imageRgb.shape[0],imageRgb.shape[1]),dtype = np.int16)
        for i in range(imageRgb.shape[0]):
            for j in range(imageRgb.shape[1]):
                B = imageRgb[i][j][0]
                G = imageRgb[i][j][1]
                R = imageRgb[i][j][2]
                imageGray[i][j] = B * self.rgb2grayCoeff[0] + G * self.rgb2grayCoeff[1] + R * self.rgb2grayCoeff[2]
        return imageGray
    def grayImageFilter(self):
        '''
        ak = cov(pI) / (σ^2 + eps) = (E(pI) - E(p)E(I)) / (σ^2 + eps)
        bk = pk - σk * uk
        qi = ak * Ii + bk
        方差是协方差的一种特殊情况，即当两个变量是相同的情况.cov(x,x) = σ^2
        '''
        #step 1
        meanI = cv2.boxFilter(self.I,-1,normalize=True,ksize=(self.radius,self.radius))
        meanp = cv2.boxFilter(self.p,-1,normalize=True,ksize=(self.radius,self.radius))
        meanIp = cv2.boxFilter(self.I * self.p,-1,normalize=True,ksize=(self.radius,self.radius))
        covIp = meanIp - meanp * meanI
        #step 2
        meanII = cv2.boxFilter(self.I * self.I,-1,normalize=True,ksize=(self.radius,self.radius))
        convII = meanII - meanI * meanI
        #step 3
        a = covIp / (convII + self.eps)
        b = meanp - a * meanI
        #step 4
        meana = cv2.boxFilter(a,-1,normalize=True,ksize=(self.radius,self.radius))
        meanb = cv2.boxFilter(b,-1,normalize=True,ksize=(self.radius,self.radius))
        #step 5
        q = meana * self.I + meanb
        return q

    def imageEnhance(self):
        pass
    def run(self):
        filteredImage = self.grayImageFilter()
        cv2.imshow("imageGray", self.I.astype(np.uint8))
        cv2.imshow("imageFilted",filteredImage.astype(np.uint8))
        cv2.imshow("diff",(self.I - filteredImage) * 255.0)
        cv2.waitKey(0)

        
        
   
class guideFilterPixlWise:
    def __init__(self,I,p,eps):
        self.I = I / 256.0
        self.p = p / 256.0
        self.eps = eps
    def imagePad(self,image,r):
        #padding 4 pixl in left and right
        rows = image.shape[0] + 2 * r
        cols = image.shape[1] + 2 * r
        imagePad = np.zeros(shape=(rows,cols),dtype=np.float16)
        #left  padding
        for row in range(r,rows - r):
            for col in range(0,r):
                imagePad[row][col] = image[row - r][col]
        #right padding
        for row in range(r,rows-r):
            for col in range(cols - r,cols):
                imagePad[row][col] = image[row - r][col - 2*r]
        #center padding
        for row in range(r,rows - r):
            for col in range(r,cols - r):
                imagePad[row][col] = image[row - r][col - r]
        #up padding
        for row in range(0,r):
            for  col in range(0,cols):
                imagePad[row][col] = imagePad[row + r][col]
        #down padding
        for row in range(rows - r,rows):
            for col in range(0,cols):
                imagePad[row][col] = imagePad[row - r][col]
        return imagePad
    def filter(self,I,p,eps):
        """        
        a = cov(I,p) / (var(I) + eps) = (E(Ip) - E(I)E(p)) / (var(I) + eps)
        b = mean(p) - a * mean(I)
        q = mean(a)I + mean(b)
        注释：
            当I的某个局部方差很小时，a = 0, b = mean(p), q = b,均值滤波
            当I的某个局部方差很大时，a = 1,b = 0, q = I,保持梯度不变
        """
        # padd 8
        #padding image
        imageIpad = np.zeros(shape=(I.shape[0] + 8,I.shape[1] + 8),dtype=np.float16)
        imagepPad = np.zeros(shape=(p.shape[0] + 8,p.shape[1] + 8),dtype=np.float16)
        meanI = np.zeros(shape=(I.shape[0],I.shape[1]),dtype=np.float16)
        meanp = np.zeros(shape=(p.shape[0],p.shape[1]),dtype=np.float16)
        meana = np.zeros(shape=(I.shape[0],I.shape[1]),dtype=np.float16)
        meanb = np.zeros(shape=(I.shape[0],I.shape[1]),dtype=np.float16)

        a = np.zeros(shape=(I.shape[0],I.shape[1]),dtype=np.float16)
        b = np.zeros(shape=(I.shape[0],I.shape[1]),dtype=np.float16)

        print("I.shape = ",I.shape)
        imageIpad = self.imagePad(I,4).astype(np.float16)
        imagepPad = self.imagePad(p,4).astype(np.float16)
        print("imagePad.shape = ",imageIpad.shape)
        # self.imshow(imageIpad.astype(np.uint8),"imagePad")
        for row in range(4,imageIpad.shape[0] - 4):
            for col in range(4,imageIpad.shape[1] - 4):
                subI = np.zeros(shape = (5,5),dtype=np.float16)
                subp = np.zeros(shape = (5,5),dtype=np.float16)
                subIp = np.zeros(shape = (5,5),dtype=np.float16)
                subII = np.zeros(shape = (5,5),dtype=np.float16)
                for r in range(-2,3):
                    for c in range(-2,3):
                        subI[r + 2,c + 2] = imageIpad[row + r,col + c]
                        subp[r + 2,c + 2] = imagepPad[row + r,col + c]
                        subIp[r + 2,c + 2] = imagepPad[row + r,col + c] * imageIpad[row + r,col + c]
                        subII[r + 2,c + 2] = imageIpad[row + r,col + c] * imageIpad[row + r,col + c]
                subMeanIp,subMeanI,subMeanp,subMeanII = 0.0,0.0,0.0,0.0
                sumIp,sumI,sump,sumII = 0.0 ,0.0,0.0,0.0
                for r in range(0,5):
                    for c in range(0,5):
                        sumIp += subIp[r,c]
                        sumI += subI[r,c]
                        sump += subp[r,c]
                        sumII += subII[r,c]
                subMeanIp = sumIp / 25
                subMeanI = sumI / 25
                subMeanp = sump / 25
                subMeanII = sumII / 25

                div = subMeanII - subMeanI * subMeanI + eps
                div = 1 if div == 0 else div
                suba = (subMeanIp - subMeanI * subMeanp) / div
                subb = subMeanp - suba * subMeanI
                a[row - 4,col - 4] = suba
                b[row - 4,col - 4] = subb
                meanp[row - 4,col - 4] = subMeanp
                meanI[row - 4,col - 4] = subMeanI
        aPad = self.imagePad(a,1)
        bPad = self.imagePad(b,1)
        for row in range(1,aPad.shape[0] - 1):
            for col in range(1,aPad.shape[1] - 1):
                suma = 0
                sumb = 0
                for r in range(-1,2):
                    for c in range(-1,2):
                        suma += aPad[row + r,col + c]
                        sumb += bPad[row + r,col + c]
                meana[row - 1,col - 1] = suma / 9
                meanb[row - 1,col - 1] = sumb / 9
        imageRes = (meana * I + meanb) * 256.0
        imageClip = np.zeros_like(imageRes,dtype=np.float16)
        for row in range(imageRes.shape[0]):
            for col in range(imageRes.shape[1]):
                tmp = imageRes[row,col]
                if tmp > 255.0:
                    tmp = 255.0
                elif tmp < 0.0:
                    tmp = 0
                imageClip[row,col] = tmp
        # self.imshow(imageClip.astype(np.uint8),"filter image")
        return imageClip.astype(np.uint8)

    def imshow(self,image,name):
        plt.figure(name)
        plt.imshow(image,cmap = 'gray')
        plt.show()

    def run(self):
        imageRes = self.filter(self.I,self.p,0.004)
        plt.figure("guide filter")
        plt.subplot(121)
        plt.imshow(self.I,cmap = "gray")
        plt.subplot(122)
        plt.imshow(imageRes,cmap = "gray")
        plt.show()

if __name__ == '__main__':
    imagePath = r"D:\software\pyproj\ImageFilter\Images\face.jpg"
    imageRgb = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    imageGray = cv2.cvtColor(imageRgb,cv2.COLOR_BGR2GRAY)
    imageGray = cv2.resize(imageGray,dsize = (128,128))
    # GuidedFilter(imageGray,imageGray,5,0.004).run()
    guideFilterPixlWise(imageGray,imageGray,0.01).run()
