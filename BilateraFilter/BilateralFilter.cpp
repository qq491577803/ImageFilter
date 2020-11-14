#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<time.h>
using namespace cv;
using namespace std;

#define sigmaColor 25
#define sigmaSpace 9
#define raidus  3
#define ImageDeep (1 << 8)
#define diam ((2 * raidus) + 1)

float clipBit(float input)
{
	float output = input > ImageDeep ? ImageDeep : input;
	output = input < 0 ? 0 : input;
	return output;
}

void buildGaussColor(float gaussColor[ImageDeep])
{
	for (int i = 0; i < ImageDeep; i++)
		gaussColor[i] = exp(- pow(i,2.0f) / (2.0f * pow(sigmaColor, 2.0f)));
}

void buildGaussSpace(float gaussSpace[diam][diam])
{
	int leftIndex = -raidus, rightIndex = raidus;
	for (int i = leftIndex; i < rightIndex + 1; i++)
		for (int j = leftIndex; j < rightIndex + 1; j++)
			gaussSpace[i - leftIndex][j - leftIndex] = exp(-(pow(i,2) + pow(j,2)) / (2.0f * pow(sigmaSpace,2.0f)));
	return;
}

cv::Mat BliateralFiter(cv::Mat srcImage)
{
	cv::Mat dstImage = srcImage.clone();
	float gaussColor[ImageDeep];
	float gaussSpace[diam][diam];
	buildGaussColor(gaussColor);
	buildGaussSpace(gaussSpace);

	int margin = raidus;
	int leftIndex = - margin;
	int rightIndex = margin;
	for (int row = margin; row < srcImage.rows - margin;row ++ )
	{
		for (int col = margin; col < srcImage.cols - margin;col ++ )
		{
			int centerValB = srcImage.at<Vec3b>(row, col)[0];
			int centerValG = srcImage.at<Vec3b>(row, col)[1];
			int centerValR = srcImage.at<Vec3b>(row, col)[2];
			float imageValB, imageValG, imageValR;
			float wtSumB, wtSumG, wtSumR;
			imageValB = imageValG = imageValR = wtSumB = wtSumG = wtSumR = 0;
			for (int i = leftIndex;i < rightIndex + 1; i++)
			{
				for	(int j = leftIndex;j < rightIndex + 1; j++)
				{
					int tmpValB = srcImage.at<Vec3b>(row - i, col - j)[0];
					int tmpValG = srcImage.at<Vec3b>(row - i, col - j)[1];
					int tmpValR = srcImage.at<Vec3b>(row - i, col - j)[2];
					const int diffB = abs(tmpValB - centerValB);
					const int diffG = abs(tmpValG - centerValG);
					const int diffR = abs(tmpValR - centerValR);
					float wtB = gaussColor[diffB] * gaussSpace[i][j];
					float wtG = gaussColor[diffG] * gaussSpace[i][j];
					float wtR = gaussColor[diffR] * gaussSpace[i][j];
					imageValB = imageValB + wtB * tmpValB;
					imageValG = imageValG + wtG * tmpValG;
					imageValR = imageValR + wtR * tmpValR;
					wtSumB += wtB;
					wtSumG += wtG;
					wtSumR += wtR;
				}
			}
			dstImage.at<Vec3b>(row, col)[0] = clipBit(int(imageValB / wtSumB));
			dstImage.at<Vec3b>(row, col)[1] = clipBit(int(imageValG / wtSumG));
			dstImage.at<Vec3b>(row, col)[2] = clipBit(int(imageValR / wtSumR));

		}
	}
	return dstImage;
}

int main()
{
	time_t startTime, endTime;
	startTime = clock();
	cv::Mat srcImage= cv::imread("E:\\ImageFilter\\Images\\faceDst.jpg");
	cv::Mat dstImage = BliateralFiter(srcImage);
	cv::imwrite("E:\\ImageFilter\\Images\\cpppdstImgage.jpg", dstImage);
	endTime = clock();
	cout << "TotalTime:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
	imshow("srcImage", srcImage);
	imshow("dstImage", dstImage);
	waitKey(0);
	system("pause");
	return 0;
}