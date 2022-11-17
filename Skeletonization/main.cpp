#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp" 
#include "opencv2\opencv.hpp"
#include <math.h>

#include <sstream>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdio.h>


using namespace std;
using namespace cv;

void on_low_h_thresh_trackbar(int, void *);
void on_high_h_thresh_trackbar(int, void *);
void on_low_s_thresh_trackbar(int, void *);
void on_high_s_thresh_trackbar(int, void *);
void on_low_v_thresh_trackbar(int, void *);
void on_high_v_thresh_trackbar(int, void *);
bool thinningIteration_fast(Mat &img, int iter);
void thinning_fast(cv::Mat& im);
void my_skelet(Mat &img1, Mat &img2);
void draw_line(cv::Mat& im, cv::Vec4i& given_line);

int low_h = 0, low_s = 0, low_v = 0;
int high_h = 179, high_s = 255, high_v = 255;


int main()
{
	string window = "original";
	string window1 = "thresh";
	string window2 = "pixels";

	namedWindow(window, CV_WINDOW_FULLSCREEN);
	namedWindow(window1, CV_WINDOW_FULLSCREEN);
	namedWindow(window2, CV_WINDOW_FULLSCREEN);


	/*createTrackbar("Low H", "original", &low_h, 179, on_low_h_thresh_trackbar);
	createTrackbar("High H", "original", &high_h, 179, on_high_h_thresh_trackbar);
	createTrackbar("Low S", "original", &low_s, 255, on_low_s_thresh_trackbar);
	createTrackbar("High S", "original", &high_s, 255, on_high_s_thresh_trackbar);
	createTrackbar("Low V", "original", &low_v, 255, on_low_v_thresh_trackbar);
	createTrackbar("High V", "original", &high_v, 255, on_high_v_thresh_trackbar);*/



	Mat img = imread("scrin.jpg");

	
	cvtColor(img, img, CV_BGR2HSV);

	
	Mat img1;
	Mat img2(480,640, CV_8UC1, Scalar(0));


	VideoCapture cap;
	bool res = cap.open("0.avi");
	if (!res) {
		throw cv::Exception(1, "error", "func", "video0.avi", 1);

	}
	
	while (1){
		Mat frame;
		Mat img;
		cap >> frame;
		cvtColor(frame, img, CV_BGR2HSV);

		inRange(img, Scalar(86, 0, 84), Scalar(121, 105, 175), img1);


		dilate(img1, img1, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
		erode(img1, img1, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

		imshow(window, img1);

		Rect rp = Rect(0, 305, 210, 175);
		Mat roi = img1(rp);

		erode(roi, roi, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)));

		Rect rp2 = Rect(0, 270, 370, 210);
		Mat roi2 = img1(rp2);


		Rect rp3 = Rect(0, 270, 370, 210);
		Mat roi3 = img2(rp3);

		roi2.copyTo(roi3);

		while (1)
		{
			Mat pixels(img2.size(), CV_8UC1, Scalar(0));
			my_skelet(img2, pixels);
			for (int i = 0; i < pixels.rows; i++)
			{
				for (int j = 0; j < pixels.cols; j++)
				{
					if ((int)pixels.at<uchar>(i, j) == 255)
					{
						img2.at<uchar>(i, j) = 0;
					}
				}

			}
			bool flag = false;
			for (int i = 0; i < pixels.rows; i++)
			{
				for (int j = 0; j < pixels.cols; j++)
				{
					if ((int)pixels.at<uchar>(i, j) == 255)
					{
						flag = true;
					}

				}
			}
			if (flag == false)
			{
				break;
			}
		}
		imshow(window1, img2);


		vector <Vec4i> lines;
		//Ищем линии по Хафу
		HoughLinesP(img2, lines, 1, CV_PI/160, 40, 40, 10000);

		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 5, CV_AA);
		}


		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l1 = lines[i];
			int Num_of_line = 0;
			int Num_of_point_1 = 0;
			int Num_of_point_2 = 0;
			int min_Len = 30000;
			for (size_t j = 0; j < lines.size(); j++)
			{
				Vec4i l2 = lines[j];
				if (i == j) break;

				int Len1 = (l1[0] - l2[0]) * (l1[0] - l2[0]) + (l1[1] - l2[1]) * (l1[1] - l2[1]);
				int Len2 = (l1[0] - l2[2]) * (l1[0] - l2[2]) + (l1[1] - l2[3]) * (l1[1] - l2[3]);
				int Len3 = (l1[2] - l2[0]) * (l1[2] - l2[0]) + (l1[3] - l2[1]) * (l1[3] - l2[1]);
				int Len4 = (l1[2] - l2[2]) * (l1[2] - l2[2]) + (l1[3] - l2[3]) * (l1[3] - l2[3]);

				if (Len1 < min_Len)
				{
					Num_of_point_1 = 0; Num_of_point_2 = 0; Num_of_line = j; min_Len = Len1;
				}
				if (Len2 < min_Len)
				{
					Num_of_point_1 = 0; Num_of_point_2 = 2; Num_of_line = j; min_Len = Len2;
				}
				if (Len3 < min_Len)
				{
					Num_of_point_1 = 2; Num_of_point_2 = 0; Num_of_line = j; min_Len = Len3;
				}
				if (Len4 < min_Len)
				{
					Num_of_point_1 = 2; Num_of_point_2 = 2; Num_of_line = j; min_Len = Len4;
				}
			}

			line(img, Point(l1[Num_of_point_1], l1[Num_of_point_1 + 1]), Point(lines[Num_of_line][Num_of_point_2], lines[Num_of_line][Num_of_point_2 + 1]), Scalar(150), 2);

		}

		cvtColor(img, img, CV_HSV2BGR);
		imshow(window2, img);
		if (waitKey(30) >= 0) break;
	}
	


	cvtColor(img, img, CV_HSV2BGR);

	waitKey(0);

	destroyAllWindows();

	return 0;


}

void on_low_h_thresh_trackbar(int, void *)
{
	low_h = min(high_h - 1, low_h);
	setTrackbarPos("Low H", "original", low_h);
}
void on_high_h_thresh_trackbar(int, void *)
{
	high_h = max(high_h, low_h + 1);
	setTrackbarPos("High H", "original", high_h);
}
void on_low_s_thresh_trackbar(int, void *)
{
	low_s = min(high_s - 1, low_s);
	setTrackbarPos("Low S", "original", low_s);
}
void on_high_s_thresh_trackbar(int, void *)
{
	high_s = max(high_s, low_s + 1);
	setTrackbarPos("High S", "original", high_s);
}
void on_low_v_thresh_trackbar(int, void *)
{
	low_v = min(high_v - 1, low_v);
	setTrackbarPos("Low V", "original", low_v);
}
void on_high_v_thresh_trackbar(int, void *)
{
	high_v = max(high_v, low_v + 1);
	setTrackbarPos("High V", "original", high_v);
}

//Шаг ускоренной скелетизации
bool thinningIteration_fast(Mat &img, int iter){
	Mat mimg(img.size(), CV_8UC1, Scalar(255));

	int maxi = img.rows - 1, maxj = img.cols - 1;
	bool diff = 0;

	//центральный бегунок
	for (int i = 1; i < maxi; i++){
		uchar* ptrs[3] = { img.ptr(i - 1), img.ptr(i), img.ptr(i + 1) };

		//+основной бегунок+
		for (int j = 1; j < maxj; j++){

			if (ptrs[1][1] == 0){
				ptrs[0]++;
				ptrs[1]++;
				ptrs[2]++;
				continue;
			}

			bool p[8];
			/*P9 P2	P3
			P8 P1	P4
			P7 P6	P5*/
			//вобьём в ядро скелетезации значения

			p[7] = ptrs[0][0] != 0;
			p[0] = ptrs[0][1] != 0;
			p[1] = ptrs[0][2] != 0;

			p[6] = ptrs[1][0] != 0;
			p[2] = ptrs[1][2] != 0;

			p[5] = ptrs[2][0] != 0;
			p[4] = ptrs[2][1] != 0;
			p[3] = ptrs[2][2] != 0;


			int tmpv = iter == 0 ? (p[0] * p[2] * p[4]) : (p[0] * p[2] * p[6]);//m1
			if (tmpv == 0){
				tmpv = iter == 0 ? (p[2] * p[4] * p[6]) : (p[0] * p[4] * p[6]);//m2
				if (tmpv == 0){
					//uchar B = 0;
					tmpv = 0; //B
					for (int ui = 0; ui < 8; ui++)
						tmpv += p[ui];

					if (tmpv >= 2 && tmpv <= 6){
						tmpv = 0; //A
						for (int ui = 0; ui < 7; ui++){
							tmpv += p[ui] == 0 && p[ui + 1] == 1;
							if (tmpv > 1)
								break;
						}
						if (tmpv <= 1){
							tmpv += p[7] == 0 && p[0] == 1;
							if (tmpv == 1){
								mimg.at<uchar>(i, j) = 0;
								//ptrs[1][1] = 0;
								if (diff == 0)diff = 1;
							}
						}
					}
				}
			}

			ptrs[0]++;
			ptrs[1]++;
			ptrs[2]++;


		}

	}

	if (diff == 1)
		multiply(img, mimg, img);

	return diff;
}

//Ускореннная скелетизация
void thinning_fast(cv::Mat& im)
{

	uchar diff;

	do {
		diff = thinningIteration_fast(im, 0);
		diff += thinningIteration_fast(im, 1);
	} while (diff > 0);


}


void my_skelet(Mat &img1, Mat &img2){

	for (int j = 1; j < img1.cols - 1; j++)
	{
		for (int i = 1; i < img1.rows - 1; i++)
		{

			if ((int)img1.at<uchar>(i, j) == 0)  //0
			{
				continue;
			}

			if ((int)img1.at<uchar>(i - 1, j) == 255 && (int)img1.at<uchar>(i, j + 1) == 255 && (int)img1.at<uchar>(i + 1, j) == 255)
			{
				continue;
			} //3

			if ((int)img1.at<uchar>(i, j + 1) == 255 && (int)img1.at<uchar>(i + 1, j) == 255 && (int)img1.at<uchar>(i, j - 1) == 255)
			{
				continue;
			}//4
			int sum = 0;

			sum = (int)img1.at<uchar>(i - 1, j - 1) + (int)img1.at<uchar>(i - 1, j) + (int)img1.at<uchar>(i - 1, j + 1) +
				(int)img1.at<uchar>(i, j - 1) + (int)img1.at<uchar>(i, j + 1) +
				(int)img1.at<uchar>(i + 1, j - 1) + (int)img1.at<uchar>(i + 1, j) + (int)img1.at<uchar>(i + 1, j + 1);

			if (sum<510 || sum>1530)
			{
				continue;
			}//1

			int jump_number = 0;
			if ((int)img1.at<uchar>(i - 1, j) == 0 && (int)img1.at<uchar>(i - 1, j + 1) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i - 1, j + 1) == 0 && (int)img1.at<uchar>(i, j + 1) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i, j + 1) == 0 && (int)img1.at<uchar>(i + 1, j + 1) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i + 1, j + 1) == 0 && (int)img1.at<uchar>(i + 1, j) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i + 1, j) == 0 && (int)img1.at<uchar>(i + 1, j - 1) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i + 1, j - 1) == 0 && (int)img1.at<uchar>(i, j - 1) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i, j - 1) == 0 && (int)img1.at<uchar>(i - 1, j - 1) == 255)
			{
				jump_number++;
			}

			if ((int)img1.at<uchar>(i - 1, j - 1) == 0 && (int)img1.at<uchar>(i - 1, j) == 255)
			{
				jump_number++;
			}

			if (jump_number != 1)
			{
				continue;
			}

			img2.at<uchar>(i, j) = 255;

		}

	}

}

