#include "reader.h"
#include "algos.h"
#include "pointspreparation.h"
#include "odometries.h"
#include "final_tests.h"
#include <vector>
#include <algorithm>
using namespace std;


int main(void)
{
	// fs::directory_iterator& src, fs::directory_iterator& masks, const string& time, const string& speed
	fs::directory_iterator src_images("D:/TRAMWAY/get.358/output_src/");
	fs::directory_iterator masks("D:/TRAMWAY/get.358/output_segm/");
	fs::directory_iterator rail_masks("D:/TRAMWAY/content/rail_marking/output/");
	//fs::directory_iterator rails_masks("D:/TRAMWAY/content/rail_marking/output/");
	const string time = "D:/TRAMWAY/get.358/Calib/frame_time.txt";
	const string speed = "D:/TRAMWAY/get.358/Calib/closest_speeds.txt";
	Reader reader(src_images, masks, rail_masks, time, speed);
	std::vector<int> dynamic_classes = {11,12,13,14,15,16,17,18 };
	float Kdata[] = { 5.8101144196059124e+02, 0., 4.6611629315197757e+02, 0.,
	   5.8101144196059124e+02, 3.1452011177827347e+02, 0., 0., 1 };
	Mat K(3, 3, cv::DataType<float>::type, Kdata);
	Vec3f R0data = { -9.6071019657095663e-02, 4.6384407919543125e-02,
	   -5.8740069123303252e-03 };
	Mat t0(Size(1, 3), CV_32FC1);
	t0.at<float>(0, 0) = 3.3300000000000002e-01;
	t0.at<float>(0, 1) = -2.0869999995529653e+00;
	t0.at<float>(0, 2) = -3.5999999999999999e-01;
	t0 = -t0; // обратный к вектору, который совмещает трамвай с камерой 
	Mat R0;
	Rodrigues(R0data, R0);
	R0.convertTo(R0, CV_32FC1);
	// ультра важный препроцессинг
	Mat rotation = Mat::eye(Size(3, 3), CV_32FC1);
	rotation.at<float>(1, 1) = 0;
	rotation.at<float>(2, 2) = 0;
	rotation.at<float>(1, 2) = 1;
	rotation.at<float>(2, 1) = -1;
	// ультра важный препроцессиг

	R0 = (R0 * rotation).inv(); // обратная к матрице, которая совмещает с.к. трамвая с с.к камеры
	Camera camera = { K, R0, t0 };
	const string prefix = "D:/TRAMWAY/get.358/results_v_diplom/";
	ofstream output_file1(prefix + "vanilla_bez_dynamic_f.txt");
	//no_optimized_odometry_on_descriptors(reader, camera, 2000, 10, dynamic_classes, output_file1, true,false);
	//ofstream output_file2(prefix + "with_cleared_masks.txt");
	//no_optimized_odometry_on_descriptors(reader, camera, 2000, 10, dynamic_classes, output_file2, true, true);
	ofstream output_file3(prefix + "optimized.txt");
	optimized_on_world_points_on_descriptors(reader, camera, 2000, 10, dynamic_classes, output_file3);

	return 0;
}