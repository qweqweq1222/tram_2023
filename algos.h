#pragma once
#include "reader.h"
#include "pointspreparation.h"
#include "optim_struct.h"
# define M_PI  3.14159265358979323846



Mat reconstruct_from_v6(double* alpha_trans)
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	double a = alpha_trans[0];
	double b = alpha_trans[1];
	double g = alpha_trans[2];
	answer.at<float>(0, 0) = cos(b) * cos(g);
	answer.at<float>(0, 1) = -sin(g) * cos(b);
	answer.at<float>(0, 2) = sin(b);
	answer.at<float>(1, 0) = sin(a) * sin(b) * cos(g) + sin(g) * cos(a);
	answer.at<float>(1, 1) = -sin(a) * sin(b) * sin(g) + cos(g) * cos(a);
	answer.at<float>(1, 2) = -sin(a) * cos(b);
	answer.at<float>(2, 0) = sin(a) * sin(g) - sin(b) * cos(a) * cos(g);
	answer.at<float>(2, 1) = sin(a) * cos(g) + sin(b) * sin(g) * cos(a);
	answer.at<float>(2, 2) = cos(a) * cos(b);
	answer.at<float>(0, 3) = alpha_trans[3];
	answer.at<float>(1, 3) = alpha_trans[4];
	answer.at<float>(2, 3) = alpha_trans[5];
	return answer;
} 
vector<double> get_angles_and_vec(const Mat Rt) // получаем углы из матрицы поворота и вектор t 
{
	double alpha, beta, gamma;
	if (abs(Rt.at<float>(0, 2)) < 1)
		beta = asin(Rt.at<float>(0, 2));
	else if (Rt.at<float>(0, 2) == 1)
		beta = M_PI / 2;
	else if (Rt.at<float>(0, 2) == -1)
		beta = -M_PI / 2;

	if (abs(Rt.at<float>(2, 2) / cos(beta)) < 1)
		alpha = acos(Rt.at<float>(2, 2) / cos(beta));
	else if (Rt.at<float>(2, 2) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(2, 2) / cos(beta) == -1)
		beta = M_PI;

	if (abs(Rt.at<float>(0, 0) / cos(beta)) < 1)
		gamma = acos(Rt.at<float>(0, 0) / cos(beta));
	else if (Rt.at<float>(0, 0) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(0, 0) / cos(beta) == -1)
		beta = M_PI;
	return { alpha, beta, gamma, Rt.at<float>(0,3), Rt.at<float>(1,3), Rt.at<float>(2,3) };
}
KeyPointMatches align_images(cv::Mat& current, cv::Mat& next, const int max_features = 5000) {

	cv::Mat im1Gray, im2Gray, descriptors1, descriptors2;
	resize(next, next, current.size());
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector< std::vector<cv::DMatch> > knn_matches;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	detector->detectAndCompute(current, cv::noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(next, cv::noArray(), keypoints2, descriptors2);
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	const float ratio_thresh = 0.5f;
	std::vector<cv::DMatch> good_matches;
	sort(knn_matches.begin(), knn_matches.end());
	double median = knn_matches[knn_matches.size() / 2].data()->distance;
	for (int i = 0; i < knn_matches.size(); ++i)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance && knn_matches[i][0].distance < median)
		{
			double dx = keypoints1[knn_matches[i][0].queryIdx].pt.x - keypoints2[knn_matches[i][0].trainIdx].pt.x;
			double dy = keypoints1[knn_matches[i][0].queryIdx].pt.y - keypoints2[knn_matches[i][0].trainIdx].pt.y;
			if (sqrt(dx * dx + dy * dy) < 100)
				good_matches.emplace_back(knn_matches[i][0]);
		}
	}
	if (good_matches.size() > max_features)
	{
		std::sort(good_matches.begin(), good_matches.end());
		good_matches.erase(good_matches.begin(), good_matches.begin() + max_features);
	}
	
	return KeyPointMatches(good_matches, keypoints1, keypoints2);
}

vector<Point> get_points_from_rails(Mat& src, Mat& rails_masks, Size& size) // трекинг точек 4-ех точек на рельсах
{
	cvtColor(rails_masks, rails_masks, COLOR_BGR2GRAY);
	cv::resize(rails_masks, rails_masks, size);
	float scale = 0.75;
	const int width = rails_masks.cols;
	const int height = rails_masks.rows;
	cout << width << endl << height << endl;
	int scale_view = scale * height;
	Mat roid = rails_masks(Range(int(scale_view), height), cv::Range(width / 4, 3 * width / 4));
	imshow("frame", roid);
	waitKey(0);
	vector<Point> rails;
	for (int y = 0; y < roid.cols; ++y)
	{
		if (int(roid.at<uchar>(roid.rows / 2, y)) == 255)
		{

			int p = y;
			Point first, second, third, fourth;
			first = { y + int(width / 4), roid.rows / 2 + int(scale_view) };
			while (int(roid.at<uchar>(roid.rows / 2, p)) == 255)
				++p;
			++p;
			second = { p + int(width / 4), roid.rows / 2 + int(scale_view) };
			while (int(roid.at<uchar>(roid.rows / 2, p + 1)) != 255)
				++p;
			third = { p + int(width / 4), roid.rows / 2 + int(scale_view) };
			++p;
			while (int(roid.at<uchar>(roid.rows / 2, p)) == 255)
				++p;
			fourth = { p + int(width / 4), roid.rows / 2 + int(scale_view) };

			rails.emplace_back((first + second) / 2);
			rails.emplace_back((third + fourth) / 2);
			break;
		}
	}
	for (int y = 0; y < roid.cols; ++y)
	{
		if (int(roid.at<uchar>(roid.rows - 1, y)) == 255)
		{
			int p = y;
			Point first, second, third, fourth;
			first = { y + int(width / 4), roid.rows - 1 + int(scale_view) };
			while (int(roid.at<uchar>(roid.rows - 1, p)) == 255)
				++p;
			++p;
			second = { p + int(width / 4), roid.rows - 1 + int(scale_view) };
			while (int(roid.at<uchar>(roid.rows - 1, p + 1)) != 255)
				++p;
			third = { p + int(width / 4), roid.rows - 1 + int(scale_view) };
			++p;
			while (int(roid.at<uchar>(roid.rows - 1, p)) == 255)
				++p;
			fourth = { p + int(width / 4), roid.rows - 1 + int(scale_view) };
			rails.emplace_back((first + second) / 2);
			rails.emplace_back((third + fourth) / 2);
			break;
		}
	}
	return rails;
}

vector<Mat> get_3d_coords(Camera& camera, vector<Point> pts2d)
{
	// перевести в СК камеры
	float fx = camera.K.at<float>(0, 0);
	float cx = camera.K.at<float>(0, 2);
	float r11 = camera.R0.at<float>(0, 0);
	float r31 = camera.R0.at<float>(2, 0);
	
	const float d = 1.54;

	float Zcu = d * (fx * r11 + cx * r31) / (pts2d[1].x - pts2d[0].x); // для верхней пары 
	float Zcl = d * (fx * r11 + cx * r31) / (pts2d[3].x - pts2d[2].x); // для нижней пары
	vector<Vec3f> pts;
	for (auto& pt2d : pts2d)
	{

	}
}



