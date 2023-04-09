#pragma once
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/flann.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <fstream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;


struct Camera {
	Mat K, R0, t0;
};

struct SimpleStruct
{
	Mat current, next, mask, rails;
	float speed;
	int dt;
};
struct KeyPointMatches
{
	vector<DMatch> matches;
	vector<KeyPoint> kp1, kp2;

	KeyPointMatches(vector<DMatch> matches_, vector<KeyPoint> kp1_,
		vector<KeyPoint> kp2_) :matches(matches_), kp1(kp1_), kp2(kp2_) {};
	~KeyPointMatches() = default;
};
class Reader {
public:
	Reader(fs::directory_iterator& src, fs::directory_iterator& masks, const string& time, const string& speed) :
		src_images(src), masks(masks)
	{
		fstream times_file(time, std::ios_base::in);
		fstream speed_file(speed, std::ios_base::in);
		int t;
		float v;
		while (times_file >> t)
			times.emplace_back(t);
		while (speed_file >> v)
			speeds.emplace_back(v);
	}

	Reader(fs::directory_iterator& src, fs::directory_iterator& masks, fs::directory_iterator& rails_masks, const string& time, const string& speed) :
		src_images(src), masks(masks), rails_masks(rails_masks)
	{
		fstream times_file(time, std::ios_base::in);
		fstream speed_file(speed, std::ios_base::in);
		int t;
		float v;
		while (times_file >> t)
			times.emplace_back(t);
		while (speed_file >> v)
			speeds.emplace_back(v);
	}

	SimpleStruct get_frame(const int& step);
	~Reader() = default;
	fs::directory_iterator src_images;
	fs::directory_iterator masks;
	fs::directory_iterator rails_masks;
	vector<int> times;
	vector<float> speeds;
	int counter = 0;
};