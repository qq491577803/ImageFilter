#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<time.h>
#include<vector>
using namespace cv;
using namespace std;

class localLaplacianFilter {

private:
	int imageWidth;
	int imageHeigh;
	// interface image
	float * srcGray; 
	float * srcRgb;
	float * dstGray;
	float * dstRgb;

	// gaussion pyramid layer
	float * gaussLayer0; // w * h
	float * gaussLayer1; // w / 2 * h / 2
	float * gaussLayer2; // w / 4 * h / 4
	float * gaussLayer3; // w / 8 * h / 8;
	float * gaussLayer4; // w / 16 * h / 16;
	// laplacian pyramid layer
	float * lapLayer0; // w * h
	float * lapLayer1; // w / 2 * h / 2
	float * lapLayer2; // w / 4 * h / 4
	float * lapLayer3; // w / 8 * h / 8;
	float * lapLayer4; // w / 16 * h / 16;
	// usimage
	float * usLayer1; // w * h
	float * usLayer2; // w / 2 * h / 2
	float * usLayer3;// w / 4 * h / 4
	float * usLayer4;// w / 8 * h / 8

	// other params
	int radius = 2;
	float kernel[5][5];
	float kernelUs[5][5];
	// private func	
	void rgb2gray(float *rgb,int w,int h);

	float remapLuma(float delta, float g0, float sigma, float beta, float alpha);
	void remapLayer(float *lapLayer, int width, int heigh, float sigma, float beta, float alpha);

	void buildGaussPyramid(); 
	void buildLaplacianPyramid();
	void rebuildLaplacianPyramid();
	void remapLaplacianPyramid();
	cv::Mat applyGainRgb();

	float* pydown(float * src, int w, int h, int radius, float kernel[][5]);
	float* pyUp(float * src, int w, int h, int radius, float kernel[][5]);
	float* imageSubstractOperator(float * src,float * usimage,int width,int heigh);

public:
	localLaplacianFilter(float *inRgb, int inWidth, int inHeigh);

	~localLaplacianFilter()
	{
		free(srcGray);
		free(srcRgb);
		free(dstGray);
		free(dstRgb);

		free(gaussLayer0);
		free(gaussLayer1);
		free(gaussLayer2);
		free(gaussLayer3);
		free(gaussLayer4);

		free(lapLayer0);
		free(lapLayer1);
		free(lapLayer2);
		free(lapLayer3);
		free(lapLayer4);

		free(usLayer1);
		free(usLayer2);
		free(usLayer3);
		free(usLayer4);
	}

	void run();
};

void writeRaw(float * image, int w, int h, string fn)
{
	string path = "./data/" + fn;
	FILE *fp = fopen(path.c_str(), "wb");
	unsigned char tmpVal = 0;
	for (int index = 0; index < (w * h); index++)
	{
		tmpVal = (unsigned char)(image[index] + 0.5f);
		fwrite(&tmpVal, sizeof(unsigned char), 1, fp);
	}
	fclose(fp);
}

void writeImage(float * image, int w, int h, string fn)
{
	string path = "./data/" + fn;
	FILE *fp = fopen(path.c_str(), "wb");

	cv::Mat dst(h, w, CV_8UC1);

	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{

			dst.at<uchar>(row, col) = (unsigned char)(image[row * w + col] + 0.5f);
		}
	}
	
	cv::imwrite(path + fn,dst);
}


void imshow(float * image,int width,int heigh,string name = "image",int channel = 1)
{
	//return;
	float tmp;
	if (channel == 1)
	{
		cv::Mat gray(heigh, width, CV_8UC1);
		for (int row = 0; row < heigh; row++)
		{
			for (int col = 0; col < width; col++)
			{
				tmp = image[row * width + col];
				tmp = tmp > 255 ? 255 : tmp;
				tmp = tmp < 0 ? 0 : tmp;
				gray.at<uchar>(row, col) = tmp;
			}
		}
	}
	else if (channel == 3)
	{
		cv::Mat gray(heigh, width, CV_8UC3);
		for (int row = 0; row < heigh; row++)
		{
			for (int col = 0; col < width; col++)
			{
				gray.at<Vec3b>(row, col)[0] = image[row * width + col];
				gray.at<Vec3b>(row, col)[1] = image[heigh * width + row * width + col];
				gray.at<Vec3b>(row, col)[2] = image[heigh * width * 2 + row * width + col];
			}
		}
		cv::imshow(name, gray);
		cv::waitKey(0);
	}else
	{
		throw "input channel num error !";
	}
}

void localLaplacianFilter::rgb2gray(float *rgb, int w, int h)
{
	srcGray = (float *)malloc(sizeof(float) * imageHeigh * imageWidth);
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			float B = rgb[row * w + col];
			float G = rgb[h * w + row * w + col];
			float R = rgb[h * w * 2 + row * w + col];
			float maxRGB = max(max(R, G), B);
			float gray = 0.25 * R + 0.5 * G + 0.25 * B;
			srcGray[row * w + col] = 0.5 * gray + maxRGB * 0.5;
		}
	}
}

localLaplacianFilter ::localLaplacianFilter(float *inRgb, int inWidth, int inHeigh)
{
	// init w,h
	this->imageWidth = inWidth;
	this->imageHeigh = inHeigh;
	// malloc resource
	srcGray = (float *)malloc(sizeof(float) * imageHeigh * imageWidth);
	srcRgb = (float *)malloc(sizeof(float) * imageHeigh * imageWidth * 3);
	dstGray = (float *)malloc(sizeof(float) * imageHeigh * imageWidth);
	dstRgb = (float *)malloc(sizeof(float) * imageHeigh * imageWidth * 3);

	// init other params
	radius = 2;
	float kernelTmp[5][5] = {
	{1,4, 6, 4, 1},
	{4,16,24,16,4},
	{6,24,36,24,6},
	{4,16,24,16,4},
	{1,4, 6, 4, 1}
	};
	for (int row = 0; row < 5; row++)
	{
		for (int col = 0; col < 5; col++)
		{
			kernel[row][col] = kernelTmp[row][col];
			kernelUs[row][col] = kernelTmp[row][col];
		}
	}

	// gauss layer
	gaussLayer0 = (float *)malloc(sizeof(float) * imageHeigh * imageWidth); 
	gaussLayer1 = (float *)malloc(sizeof(float) * imageHeigh / 2 * imageWidth / 2); 
	gaussLayer2 = (float *)malloc(sizeof(float) * imageHeigh / 4 * imageWidth / 4); 
	gaussLayer3 = (float *)malloc(sizeof(float) * imageHeigh / 8 * imageWidth / 8); 
	gaussLayer4 = (float *)malloc(sizeof(float) * imageHeigh / 16 * imageWidth / 16);
	// laplacian layer
	lapLayer0 = (float *)malloc(sizeof(float) * imageHeigh * imageWidth);
	lapLayer1 = (float *)malloc(sizeof(float) * imageHeigh / 2 * imageWidth / 2);
	lapLayer2 = (float *)malloc(sizeof(float) * imageHeigh / 4 * imageWidth / 4);
	lapLayer3 = (float *)malloc(sizeof(float) * imageHeigh / 8 * imageWidth / 8);
	lapLayer4 = (float *)malloc(sizeof(float) * imageHeigh / 16 * imageWidth / 16);

	// usimage
	usLayer1 = (float *)malloc(sizeof(float) * imageHeigh * imageWidth);
	usLayer2 = (float *)malloc(sizeof(float) * imageHeigh / 2 * imageWidth / 2);
	usLayer3 = (float *)malloc(sizeof(float) * imageHeigh / 4 * imageWidth / 4);
	usLayer4 = (float *)malloc(sizeof(float) * imageHeigh / 8 * imageWidth / 8);

	// init srcRgb
	memcpy(srcRgb, inRgb, sizeof(float) * imageHeigh * imageWidth * 3);

	// rgb2y
	rgb2gray(srcRgb,imageWidth,imageHeigh);
	imshow(srcGray, imageWidth, imageHeigh, "srcGray", 1);
	writeRaw(srcGray, imageWidth, imageHeigh, "srcGray.raw");
	//writeImage(srcGray, imageWidth, imageHeigh, "srcGray.jpg");
}

float * imagePadding(float *srcImage,int width,int heigh,int radius)
{
	float *imagePad = (float *)malloc(sizeof(float) * (width + 2 * radius) * (heigh + 2 * radius));	
	// center region
	for (int row = radius; row < heigh + radius; row++)
	{
		for (int col = radius; col < width + radius; col++)
		{
			imagePad[row * (width + 2 * radius) + col] = srcImage[(row - radius) * width + col - radius];
		}
	}
	
	// top region
	for (int row = 0; row < radius; row++)
	{
		for (int col = radius; col < width + radius; col++)
		{
			imagePad[row * (width + 2 * radius) + col] = imagePad[radius * (width + 2 * radius) + col];
		}
	}

	// bottom region
	for (int row = (heigh + radius); row < (heigh + 2 * radius); row++)
	{
		for (int col = radius; col < width + radius; col++)
		{
			imagePad[row * (width + 2 * radius) + col] = imagePad[(heigh + radius - 1) * (width + 2 * radius) + col];
		}
	}

	// left region
	for (int row = 0; row < heigh + 2 * radius; row++)
	{
		for (int col = 0; col < radius; col++)
		{
			imagePad[row * (width + 2 * radius) + col] = imagePad[row * (width + 2 * radius) + radius];
		}
	}

	// right region
	for (int row = 0; row < heigh + 2 * radius; row++)
	{
		for (int col = width + radius; col < width + 2 * radius; col++)
		{
			imagePad[row * (width + 2 * radius) + col] = imagePad[row * (width + 2 * radius) + width + radius - 1];
		}
	}
	return imagePad;
}

float * imageConv2D(float *imagePad,int width,int heigh,int radius,float kernel[][5])
{
	float *lpfImage = (float *)malloc(sizeof(float) * width * heigh);
	
	int padWidth = width + 2 * radius;
	int padHeigh = heigh + 2 * radius;

	for (int row = radius; row < heigh + radius; row++)
	{
		for (int col = radius; col < width + radius; col++)
		{
			float vsum = 0.0;
			float pcnt = 0.0;
			for (int dx = -radius; dx <= radius; dx++)
			{
				for (int dy = -radius; dy <= radius; dy++)
				{			
					vsum += imagePad[(row + dy) * padWidth + col + dx] * kernel[radius + dy][radius + dx];
					pcnt += 1 * kernel[radius + dy][radius + dx];
				}
			}

			if (pcnt > 0.0)
			{
				lpfImage[(row - radius) * width + col - radius] = vsum / pcnt;
			}
			else {
				lpfImage[(row - radius) * width + col - radius] = imagePad[row * padWidth + col];
			}
		}
	}
	return lpfImage;
}

float * imageConv2DUs(float *imagePad, int width, int heigh, int radius, float kernel[][5])
{
	float *lpfImage = (float *)malloc(sizeof(float) * width * heigh);

	int padWidth = width + 2 * radius;
	int padHeigh = heigh + 2 * radius;

	for (int row = radius; row < heigh + radius; row++)
	{
		for (int col = radius; col < width + radius; col++)
		{
			float vsum = 0.0;
			float pcnt = 0.0;
			for (int dx = -radius; dx <= radius; dx++)
			{
				for (int dy = -radius; dy <= radius; dy++)
				{
					vsum += imagePad[(row + dy) * padWidth + col + dx] * kernel[radius + dy][radius + dx];
					pcnt += 1 * kernel[radius + dy][radius + dx];
				}
			}

			if (pcnt > 0.0)
			{
				lpfImage[(row - radius) * width + col - radius] = vsum / pcnt * 4;
			}
			else {
				lpfImage[(row - radius) * width + col - radius] = imagePad[row * padWidth + col];
			}
		}
	}
	return lpfImage;
}

float * imageLowPassFilter(float *srcImage,int width,int heigh,int radius,float kernel[][5])
{
	float *imagePad = imagePadding(srcImage, width, heigh, radius);
	float *lpfImage = imageConv2D(imagePad, width, heigh, radius, kernel);	 

	free(imagePad);
	return lpfImage;
}

float * imageDownSample(float * srcImage,int width,int heigh)
{
	if (width % 2 != 0 || heigh % 2 != 0)
	{
		printf("src width,heigh = %d,%d,width and heigh must be divided by 2 ! \n",width,heigh);
		throw "error";
		return NULL;
	}
	float *dsImage = (float *)malloc(sizeof(float) * (width / 2) * (heigh / 2));
	
	int xRow = 0;
	for (int row = 0; row < heigh / 2; row++)
	{
		int yCol = 0;
		for (int col = 0; col < width / 2; col++)
		{
			yCol = yCol > (width - 1) ? (width - 1) : yCol;
			xRow = xRow > (heigh - 1) ? (heigh - 1) : xRow;

			dsImage[row * (width / 2) + col] = srcImage[xRow * width + yCol];
			yCol += 2;			
		}
		xRow += 2;
	}
	return dsImage;
}


float* localLaplacianFilter::pydown(float * src, int width, int heigh, int radius, float kernel[][5])
{
	float *lpfImage = imageLowPassFilter(src, width, heigh, radius, kernel);
	float *dsImage = imageDownSample(lpfImage, width, heigh);
	free(lpfImage);
	return dsImage;
}

float* localLaplacianFilter::pyUp(float * src, int width, int heigh, int radius, float kernel[][5])
{
	float *interpImage = (float *)malloc(sizeof(float) * 2 * width * 2 * heigh);
	
	int usW = width * 2;
	int usH = heigh * 2;
	// initi interpImage
	for (int row = 0; row < usH; row++)
	{
		for (int col = 0; col < usW; col++)
		{
			interpImage[row * usW + col] = 0.0;
		}
	}
	
	int xRow = 0;
	for (int row = 0; row < usH; row+= 2)
	{
		int yCol = 0;
		for (int col = 0; col < usW; col+= 2)
		{
			interpImage[row * usW + col] = src[xRow * width + yCol];	
			yCol += 1;
		}
		xRow += 1;
	}

	float *imagePad = imagePadding(interpImage, usW, usH, radius);
	float *usImage = imageConv2DUs(imagePad, usW, usH, radius, kernel);

	free(imagePad);
	free(interpImage);
	return usImage;
}


void localLaplacianFilter::buildGaussPyramid()
{
	// gaussLayer0
	memcpy(gaussLayer0,srcGray,sizeof(float) * imageWidth * imageHeigh);
	// gaussLayer1
	gaussLayer1 = pydown(srcGray, imageWidth, imageHeigh,radius,kernel);
	// gaussLayer2
	gaussLayer2 = pydown(gaussLayer1, imageWidth / 2, imageHeigh / 2, radius, kernel);
	// gaussLayer3
	gaussLayer3 = pydown(gaussLayer2, imageWidth / 4, imageHeigh / 4, radius, kernel);
	// gaussLayer4
	gaussLayer4 = pydown(gaussLayer3, imageWidth / 8, imageHeigh / 8, radius, kernel);
	
	if (0)
	{
		imshow(gaussLayer0, imageWidth, imageHeigh, "gaussLayer0", 1);
		imshow(gaussLayer1, imageWidth / 2, imageHeigh / 2, "gaussLayer1", 1);
		imshow(gaussLayer2, imageWidth / 4, imageHeigh / 4, "gaussLayer2", 1);
		imshow(gaussLayer3, imageWidth / 8, imageHeigh / 8, "gaussLayer3", 1);
		imshow(gaussLayer4, imageWidth / 16, imageHeigh / 16, "gaussLayer4", 1);
	}
}

float *localLaplacianFilter::imageSubstractOperator(float * src, float * usimage, int width, int heigh)
{
	float res;
	float *residual = (float*)malloc(sizeof(float) * width * heigh);

	for (int row = 0; row < heigh; row++)
	{
		for (int col = 0; col < width; col++)
		{
			res = src[row * width + col] - usimage[row * width + col];
			res = res > 255 ? 255 : res;
			res = res < -255 ? -255 : res;
			residual[row * width + col] = res;
		}
	}
	return residual;
}

void localLaplacianFilter::buildLaplacianPyramid()
{
	usLayer1 = pyUp(gaussLayer1, imageWidth / 2, imageHeigh / 2, 2, kernelUs);
	usLayer2 = pyUp(gaussLayer2, imageWidth / 4, imageHeigh / 4, 2, kernelUs);
	usLayer3 = pyUp(gaussLayer3, imageWidth / 8, imageHeigh / 8, 2, kernelUs);
	usLayer4 = pyUp(gaussLayer4, imageWidth / 16, imageHeigh / 16, 2, kernelUs);

	lapLayer0 = imageSubstractOperator(gaussLayer0, usLayer1, imageWidth, imageHeigh);
	lapLayer1 = imageSubstractOperator(gaussLayer1, usLayer2, imageWidth / 2, imageHeigh / 2);
	lapLayer2 = imageSubstractOperator(gaussLayer2, usLayer3, imageWidth / 4, imageHeigh / 4);
	lapLayer3 = imageSubstractOperator(gaussLayer3, usLayer4, imageWidth / 8, imageHeigh / 8);
	memcpy(lapLayer4,gaussLayer4,imageWidth / 16 * imageHeigh / 16 * sizeof(float));

	if (0)
	{
		imshow(usLayer1, imageWidth , imageHeigh, "usLayer1", 1);
		imshow(usLayer2, imageWidth / 2, imageHeigh / 2, "usLayer2", 1);
		imshow(usLayer3, imageWidth / 4, imageHeigh / 4, "usLayer3", 1);
		imshow(usLayer4, imageWidth / 8, imageHeigh / 8, "usLayer4", 1);
	}
	if (0)
	{		
		imshow(lapLayer0, imageWidth, imageHeigh, "lapLayer0", 1);
		imshow(lapLayer1, imageWidth / 2, imageHeigh / 2, "lapLayer1", 1);
		imshow(lapLayer2, imageWidth / 4, imageHeigh / 4, "lapLayer2", 1);
		imshow(lapLayer3, imageWidth / 8, imageHeigh / 8, "lapLayer3", 1);
		imshow(lapLayer4, imageWidth / 16, imageHeigh / 16, "lapLayer4", 1);
	}	
}

float* imageAddOperator(float * src1,float *src2,int width,int heigh)
{
	float *resImage = (float*)malloc(width * heigh * sizeof(float));
	float tmp;
	float lap;
	for (int row = 0; row < heigh; row++)
	{
		for (int col = 0; col < width; col++)
		{
			lap = src2[row * width + col];
			tmp = src1[row * width + col] + lap;
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			resImage[row * width + col] = tmp;
		}		
	}
	return resImage;
}

void localLaplacianFilter::rebuildLaplacianPyramid()
{
	float *tmpImage1;
	float *tmpImage2;
	tmpImage1 = pyUp(lapLayer4, imageWidth / 16, imageHeigh / 16, 2, kernelUs);
	
	tmpImage2 = imageAddOperator(tmpImage1,lapLayer3, imageWidth / 8, imageHeigh / 8);
	free(tmpImage1);
	tmpImage1 = pyUp(tmpImage2, imageWidth / 8, imageHeigh / 8, 2, kernelUs);
	free(tmpImage2);

	tmpImage2 = imageAddOperator(tmpImage1, lapLayer2, imageWidth / 4, imageHeigh / 4);
	free(tmpImage1);
	tmpImage1 = pyUp(tmpImage2, imageWidth / 4, imageHeigh / 4, 2, kernelUs);
	free(tmpImage2);

	tmpImage2 = imageAddOperator(tmpImage1, lapLayer1, imageWidth / 2, imageHeigh / 2);
	free(tmpImage1);
	tmpImage1 = pyUp(tmpImage2, imageWidth / 2, imageHeigh / 2, 2, kernelUs);
	free(tmpImage2);

	tmpImage2 = imageAddOperator(tmpImage1, lapLayer0, imageWidth, imageHeigh);
	dstGray = tmpImage2;
	//writeImage(dstGray, imageWidth, imageHeigh, "dstGray.jpg");
}

inline float localLaplacianFilter::remapLuma(float delta,float g0,float sigma, float beta,float alpha)
{
	float diffAbs = abs(delta);
	int signV = delta >= 0.0 ? 1 : -1;
	if (diffAbs > sigma)
	{
		return (g0 + signV * ((diffAbs - sigma) * beta + sigma));
	}
	else
	{
		return (g0 + signV * pow(diffAbs / sigma, alpha) * sigma);
	}
}

void localLaplacianFilter::remapLayer(float *lapLayer,int width,int heigh,float sigma,float beta,float alpha)
{
	float tmpVal;
	for (int row = 0; row < heigh; row++)
	{
		for (int col = 0; col < width; col++)
		{
			tmpVal = lapLayer[row * width + col];
			tmpVal = remapLuma(tmpVal, 0, sigma, beta, alpha);
			lapLayer[row * width + col] = tmpVal;// *3.5;
		}
	}
}

void localLaplacianFilter::remapLaplacianPyramid()
{
	float sigma, beta, alpha;

	// layer0  w * h   lapLayer0
	//sigma = 50, beta = 1, alpha = 0.5;
	//remapLayer(lapLayer0, imageWidth, imageHeigh, sigma, beta, alpha );
	// layer1  w / 2 * h / 2  lapLayer1
	sigma = 40, beta = 1, alpha = 0.5;
	remapLayer(lapLayer1, imageWidth / 2, imageHeigh / 2, sigma, beta, alpha);

	// layer2   w / 4 * h / 4  lapLayer2;
	sigma = 30, beta = 1, alpha = 0.5;
	remapLayer(lapLayer2, imageWidth / 4, imageHeigh / 4, sigma, beta, alpha);
	// layer3   w / 8 * h / 8  lapLayer3; 
	sigma = 30, beta = 1, alpha = 0.5;
	remapLayer(lapLayer3, imageWidth / 8, imageHeigh / 8, sigma, beta, alpha);
	// layer4   w / 16 * h / 16 lapLayer4; 
	//sigma = 1, beta = 1, alpha = 0.5;
	//remapLayer(lapLayer4, imageWidth / 16, imageHeigh / 16, sigma, beta, alpha);
}

cv::Mat localLaplacianFilter::applyGainRgb()
{
	cv::Mat rgbEnhance(imageHeigh, imageWidth, CV_8UC3);
	for (int row = 0; row < imageHeigh; row++)
	{
		for (int col = 0; col < imageWidth; col++)
		{
			float src = srcGray[row * imageWidth + col];
			float dst = dstGray[row * imageWidth + col];
			float gain = src == 0 ? 1.0 : (dst / src);

			float B = srcRgb[row * imageWidth + col] * gain;
			float G = srcRgb[imageHeigh * imageWidth + row * imageWidth + col] * gain;
			float R = srcRgb[imageHeigh * imageWidth * 2 + row * imageWidth + col] * gain;

			B = B > 255 ? 255 : B;
			B = B < 0 ? 0 : B;
			G = G > 255 ? 255 : G;
			G = G < 0 ? 0 : G;
			R = R > 255 ? 255 : R;
			R = R < 0 ? 0 : R;
			rgbEnhance.at<Vec3b>(row, col)[0] = B;
			rgbEnhance.at<Vec3b>(row, col)[1] = G;
			rgbEnhance.at<Vec3b>(row, col)[2] = R;
		}
	}
	return rgbEnhance;                                                                                                                                                                                                                                                    
}

void rgb2raw(cv::Mat rgb)
{
	char path[256] = "./data/face.raw";
	int imageWidth = rgb.cols;
	int imageHeigh = rgb.rows;
	unsigned char * rawBuffer = (unsigned char *)malloc(sizeof(unsigned char) * imageHeigh * imageWidth * 3);
	FILE *fpRgb = fopen(path, "wb");
	int indexx = 0;
	for(int row = 0;row < imageHeigh;row ++)
		for(int col = 0;col < imageWidth;col ++)
		{
			for (int c = 0; c < rgb.channels(); c++)
			{
				indexx = imageWidth * imageHeigh * c;
				rawBuffer[indexx + row * imageWidth + col] = (unsigned char)rgb.at<Vec3b>(row, col)[c];
	
			}
		}
	fwrite(rawBuffer,sizeof(unsigned char) * imageHeigh * imageWidth * 3,1,fpRgb);
	fclose(fpRgb);
}


void readRgbImage(float * rgb,int w,int h)
{
	char path[256] = "./data/face.raw";
	int imageWidth = w;
	int imageHeigh = h;
	unsigned char * rawBuffer = (unsigned char *)malloc(sizeof(unsigned char) * imageHeigh * imageWidth * 3);
	FILE *fpRgb = fopen(path, "rb");
	fread(rawBuffer,sizeof(unsigned char) * imageWidth * imageHeigh * 3,1, fpRgb);
	for (int index = 0; index < imageWidth * imageHeigh * 3; index++)
	{
		rgb[index] = (float)rawBuffer[index];
	}
}

void localLaplacianFilter::run()
{
	buildGaussPyramid();
	buildLaplacianPyramid();
	remapLaplacianPyramid();
	rebuildLaplacianPyramid();

	cv::Mat rgbEnhance(imageHeigh, imageWidth, CV_8UC3);
	rgbEnhance = applyGainRgb();
	cv::imwrite("./data/dxoMark/dstRgbgallery1.jpg", rgbEnhance);
}

void llfRun()
{
	cv::Mat srcImage = cv::imread("./data/dxoMark/gallery1.jpeg", IMREAD_UNCHANGED);
	//cv::resize(srcImage,srcImage,cv::Size(4096,3072));
	//cv::imwrite("./data/dxoMark/building.jpg",srcImage);
	printf("srcImage width = %d,heigh = %d\n",srcImage.cols,srcImage.rows);	

	rgb2raw(srcImage);
	float *srcRgb = (float *)malloc(sizeof(float) * srcImage.cols * srcImage.rows * 3);

	readRgbImage(srcRgb, srcImage.cols, srcImage.rows);

	localLaplacianFilter laplacianFilter(srcRgb, srcImage.cols, srcImage.rows);
	laplacianFilter.run();
}
