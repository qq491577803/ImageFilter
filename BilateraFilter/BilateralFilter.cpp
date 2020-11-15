#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<time.h>
#include<vector>
using namespace cv;
using namespace std;

#define sigmaColor 25
#define sigmaSpace 9
#define raidus  3
#define ImageDeep ((1 << 8) - 1)
#define diam ((2 * raidus) + 1)

int clipBit(int input)
{
	int output = input > ImageDeep ? ImageDeep : input;
	output = output < 0 ? 0 : output;
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
			if (srcImage.channels() == 3) 
			{
				int centerValB = srcImage.at<Vec3b>(row, col)[0];
				int centerValG = srcImage.at<Vec3b>(row, col)[1];
				int centerValR = srcImage.at<Vec3b>(row, col)[2];
				float imageValB, imageValG, imageValR;
				float wtSumB, wtSumG, wtSumR;
				imageValB = imageValG = imageValR = wtSumB = wtSumG = wtSumR = 0;
				for (int i = leftIndex; i < rightIndex + 1; i++)
				{
					for (int j = leftIndex; j < rightIndex + 1; j++)
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
			else
			{
				int centerValB = srcImage.at<uchar>(row, col);
				float imageValB = 0;
				float wtSumB = 0;
				for (int i = leftIndex; i < rightIndex + 1; i++)
				{
					for (int j = leftIndex; j < rightIndex + 1; j++)
					{
						int tmpValB = srcImage.at<uchar>(row - i, col - j);
						const int diffB = abs(tmpValB - centerValB);
						float wtB = gaussColor[diffB] * gaussSpace[i][j];
						imageValB = imageValB + wtB * tmpValB;
						wtSumB += wtB;
					}
				}
				dstImage.at<uchar>(row, col) = clipBit(int(imageValB / wtSumB));
			}
		}
	}
	return dstImage;
}

cv::Mat rgb2gray(cv::Mat srcRGB)
{
	cv::Mat dstGray(srcRGB.rows,srcRGB.cols,CV_8UC1);
	float coeff[3] = {0.0722,0.7152,0.2126};
	for (int row = 0; row < srcRGB.rows; row++)
		for (int col = 0; col < srcRGB.cols; col++) 
			dstGray.at<uchar>(row, col) = srcRGB.at<Vec3b>(row, col)[0] * coeff[0] +
				srcRGB.at<Vec3b>(row, col)[1] * coeff[1] +
				srcRGB.at<Vec3b>(row, col)[2] * coeff[2];
	return dstGray;
}

cv::Mat imageEnhance(cv::Mat Image)
{
	cv::Mat srcRGB = Image.clone();
	cv::Mat srcGray = rgb2gray(srcRGB);
	cv::Mat grayFilter = BliateralFiter(srcGray);
	cv::Mat grayEnhanced(grayFilter.rows,grayFilter.cols,CV_32FC1);
#define TMS 2.5f
	//enhance Gray image
	cv::Mat resEnhancedGray(srcGray.rows,srcGray.cols,CV_8UC1);
	for (int row = 0; row < grayFilter.rows; row++)
		for (int col = 0; col < grayFilter.cols; col++)
		{
			grayEnhanced.at<float>(row, col) = (float)(srcGray.at<uchar>(row, col)) + TMS * ((float)(srcGray.at<uchar>(row, col)) - (float)grayFilter.at<uchar>(row, col));
			resEnhancedGray.at<uchar>(row, col) = clipBit((int)grayEnhanced.at<float>(row, col));
		}
	cv::imwrite("E:\\ImageFilter\\Images\\srcGray.jpg", srcGray);
	cv::imwrite("E:\\ImageFilter\\Images\\grayFilter.jpg", grayFilter);
	cv::imwrite("E:\\ImageFilter\\Images\\resEnhancedGray.jpg", resEnhancedGray);
	cv::imshow("srcGray", srcGray);
	cv::imshow("grayFilter", grayFilter);
	cv::imshow("resEnhancedGray", resEnhancedGray);
	cv::waitKey(0);
	//Apply gain to RGB
	cv::Mat resEnhancedRGB = Image.clone();
	for(int row = 0; row < resEnhancedGray.rows;row++)
		for(int col = 0; col < resEnhancedGray.cols;col++)
			for (int ch = 0; ch < 3; ch++)
			{
				float tmp =((float)resEnhancedGray.at<uchar>(row, col) / (float)srcGray.at<uchar>(row, col));
				tmp = resEnhancedRGB.at<Vec3b>(row, col)[ch] * tmp;
				resEnhancedRGB.at<Vec3b>(row, col)[ch] = clipBit((int)tmp);
			}
	cv::imwrite("E:\\ImageFilter\\Images\\srcImage.jpg", Image);
	cv::imwrite("E:\\ImageFilter\\Images\\resEnhancedRGB.jpg", resEnhancedRGB);
	cout << "srcshape" << Image.rows << "x" << Image.cols << endl;
	cout << "dstshape" << resEnhancedRGB.rows << "x" << resEnhancedRGB.cols << endl;
	cv::imshow("resEnhancedRGB", resEnhancedRGB);
	cv::imshow("srcImage", Image);
	cv::waitKey(0);
	return Image;
}
int main()
{
	time_t startTime, endTime;
	startTime = clock();
	cv::Mat srcImage = cv::imread("E:\\ImageFilter\\Images\\face.jpg",IMREAD_UNCHANGED);
	imageEnhance(srcImage);//BilateralFilter Image Enhance.
	endTime = clock();
	cout << "TotalTime:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
	return 0;
}