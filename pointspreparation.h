#pragma once
#include "reader.h"
Mat estimate(Point pt, Mat& R0, Mat& t0, Mat& K) // оценка точек в ск камеры
{
	Mat Rt = Mat::eye(Size(4, 3), CV_32FC1);

	R0.copyTo(Rt(Rect(0, 0, 3, 3)));
	t0.copyTo(Rt(Rect(3, 0, 1, 3)));

	Mat P = K * Rt;
	Mat P_roi = Mat::eye(Size(3, 3), CV_32FC1);

	P(Rect(0, 0, 2, 3)).copyTo(P_roi(Rect(0, 0, 2, 3)));
	P(Rect(3, 0, 1, 3)).copyTo(P_roi(Rect(2, 0, 1, 3)));

	Mat answer(Size(1, 3), CV_32FC1);
	answer.at<float>(0, 0) = pt.x;
	answer.at<float>(1, 0) = pt.y;
	answer.at<float>(2, 0) = 1;
	Mat pt3d = P_roi.inv() * answer;
	pt3d /= pt3d.at<float>(2);

	Mat pt3d_camera = Mat(Size(1, 4), CV_32FC1);

	pt3d_camera.at<float>(0, 0) = pt3d.at<float>(0, 0);
	pt3d_camera.at<float>(1, 0) = pt3d.at<float>(1, 0);
	pt3d_camera.at<float>(2, 0) = 0;
	pt3d_camera.at<float>(3, 0) = 1;

	return Rt * pt3d_camera;
}

Mat general_estimate(Point pt, Mat& R0, Mat& t0, Mat& K, Mat& plane) // общая оценка точки, лежащий на плоскости Ax + By + Cz + D = 0, которая задется Mat plane в СК робота
{
	Mat Rt = Mat::eye(Size(4, 3), CV_32FC1);

	R0.copyTo(Rt(Rect(0, 0, 3, 3)));
	t0.copyTo(Rt(Rect(3, 0, 1, 3)));

	Mat P = K * Rt;
	Mat P_abcd = Mat::eye(Size(4, 4), CV_32FC1);
	P.copyTo(P_abcd(Rect(0, 0, 4, 3)));
	plane.copyTo(P_abcd(Rect(0, 3, 4, 1)));


	Mat answer(Size(1, 4), CV_32FC1);
	answer.at<float>(0, 0) = pt.x;
	answer.at<float>(1, 0) = pt.y;
	answer.at<float>(2, 0) = 1;
	answer.at<float>(3, 0) = 0;
	Mat pt3d = P_abcd.inv() * answer;
	pt3d /= pt3d.at<float>(3);

	return pt3d;
}
bool is_dynamic(Mat& frame, Mat mask, Point pt, vector<int>& dynamic_classes, const int kernel = 11) // проверяем, чтобы в окрестности точек не было динамических объектов
{
	resize(mask, mask, frame.size()); // маски другого размера по дефолту
	int xl = pt.x - round(kernel / 2) < 0 ? 0 : pt.x - round(kernel / 2);
	int yl = pt.y - round(kernel / 2) < 0 ? 0 : pt.y - round(kernel / 2);
	int xu = pt.x + round(kernel / 2) > (frame.cols - 1) ? (frame.cols - 1) : pt.x + round(kernel / 2);
	int yu = pt.y + round(kernel / 2) > (frame.rows - 1) ? (frame.rows - 1) : pt.y + round(kernel / 2);

	for (int i = xl; i < xu; ++i)
		for (int j = yl; j < yu; ++j)
		{
			if (find(dynamic_classes.begin(), dynamic_classes.end(), int(mask.at<uchar>(j, i))) != dynamic_classes.end())
				return true;
		}

	return false;
}