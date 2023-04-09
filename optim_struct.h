#pragma once
#include "reader.h"
struct SnavelyReprojectionError {
	SnavelyReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy, int width, int height)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), width(width), height(height) {}

	template <typename T>
	bool operator()(const T* const alpha_t, const T* const pt3d, T* residuals) const {


		T P3[3];
		T a = alpha_t[0];
		T b = alpha_t[1];
		T g = alpha_t[2];

		P3[0] = T(cos(b) * cos(g)) * (pt3d[0]) - T(sin(g) * cos(b)) * (pt3d[1]) + T(sin(b)) * (pt3d[2]) + alpha_t[3];
		P3[1] = T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * (pt3d[0]) + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * (pt3d[1]) - T(sin(a) * cos(b)) * (pt3d[2]) + alpha_t[4];
		P3[2] = T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * (pt3d[0]) + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * (pt3d[1]) + T(cos(a) * cos(b)) * (pt3d[2]) + alpha_t[5];

		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);

		residuals[0] = (predicted_x - T(observed_x)) / T(50);
		residuals[1] = (predicted_y - T(observed_y)) / T(50);
		residuals[2] = predicted_x > T(0) && predicted_x < T(width) ? T(0) : T(1000);
		residuals[3] = predicted_y > T(0) && predicted_y < T(height) ? T(0) : T(1000);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy,
		const int width, const int height) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 4, 6, 3>(
			new SnavelyReprojectionError(observed_x, observed_y, fx, fy, cx, cy, width, height)));
	}

	double observed_x;
	double observed_y;
	int width, height;
	double fx, fy, cx, cy;
};


struct SnavelyReprojectionErrorWorld { // здесь уже мы передаем положение точки в С.К. робота и поэтому необходимо передавать дополнительно R0,t0 
	SnavelyReprojectionErrorWorld(double observed_x, double observed_y, double fx, double fy, double cx, double cy, cv::Mat R0, cv::Mat t0)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), R0(R0), t0(t0) {}

	template <typename T>
	bool operator()(const T* const alpha_t, const T* const pt3d, T* residuals) const {


		T P3[3];
		T a = alpha_t[0];
		T b = alpha_t[1];
		T g = alpha_t[2];

		// переходим в с.к. камеры R0 * r_world + t0 = r_camera
		P3[0] = T(R0.at<float>(0,0))* pt3d[0] + T(R0.at<float>(0,1))* pt3d[1] + T(R0.at<float>(0,2))* pt3d[2] + T(t0.at<float>(0,0));
		P3[1] = T(R0.at<float>(1, 0)) * pt3d[0] + T(R0.at<float>(1, 1)) * pt3d[1] + T(R0.at<float>(1, 2)) * pt3d[2] + T(t0.at<float>(1, 0));
		P3[2] = T(R0.at<float>(2, 0)) * pt3d[0] + T(R0.at<float>(2, 1)) * pt3d[1] + T(R0.at<float>(2, 2)) * pt3d[2] + T(t0.at<float>(2, 0));
		// от с.к. камеры переходим к проекции K*(R*r_camera + t) 
		P3[0] = T(cos(b) * cos(g)) * (P3[0]) - T(sin(g) * cos(b)) * (P3[1]) + T(sin(b)) * (P3[2]) + alpha_t[3];
		P3[1] = T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * (P3[0]) + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * (P3[1]) - T(sin(a) * cos(b)) * (P3[2]) + alpha_t[4];
		P3[2] = T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * (P3[0]) + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * (P3[1]) + T(cos(a) * cos(b)) * (P3[2]) + alpha_t[5];

		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);
		// определяем ошибку
		residuals[0] = (predicted_x - T(observed_x));
		residuals[1] = (predicted_y - T(observed_y));
		residuals[2] = alpha_t[4] * T(10000);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const cv::Mat R0, const cv::Mat t0) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWorld, 3, 6, 3>(
			new SnavelyReprojectionErrorWorld(observed_x, observed_y, fx, fy, cx, cy, R0, t0)));
	}

	double observed_x;
	double observed_y;
	double fx, fy, cx, cy;
	Mat R0, t0;
};