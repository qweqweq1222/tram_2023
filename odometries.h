#pragma once
#include "algos.h"

// наивная одометрия без оптимизации, но с сегментацией динамических объектов

void vanilla_odometry(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file)
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1);
	SimpleStruct frame;
	for (int k = 0; k < number_of_iterations; k += step)
	{
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		KeyPointMatches kpm = align_images(frame.current, frame.next, 500);
		vector<Point2f> kp1, kp2;
		for (auto& match : kpm.matches)
		{
			float u = (float(kpm.kp1.at(match.queryIdx).pt.x));
			float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
			if (dynamic_classes.size() != 0) // если точка динамическая, то сразу выкидываем
				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(v), int(u)))) != dynamic_classes.end())
					continue;
			
			if (!is_dynamic(frame.current, frame.mask, Point(u, v), dynamic_classes) && dynamic_classes.size() != 0) // точка не динамическая - проверяем окрестность 
			{
				kp1.push_back({ u,v });
				kp2.push_back({ float(kpm.kp2.at(match.trainIdx).pt.x) , float(kpm.kp2.at(match.trainIdx).pt.y) });
			}
			else if (dynamic_classes.empty()) // вообще без масок работаем - все добавляем подряд
			{
				kp1.push_back({ u,v });
				kp2.push_back({ float(kpm.kp2.at(match.trainIdx).pt.x) , float(kpm.kp2.at(match.trainIdx).pt.y) });
			}
		}
		Mat E, R, t, useless_masks;
		E = findEssentialMat(kp1, kp2, camera.K, RANSAC, 0.99, 1.0, useless_masks);
		recoverPose(E, kp1, kp2, camera.K, R, t, useless_masks);

		t.convertTo(t, CV_32FC1);
		R.convertTo(R, CV_32FC1);

		R = camera.R0.inv() * R * camera.R0;
		t = camera.R0.inv() * t;
		t *= (0.001 * frame.dt * frame.speed);

		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		GLOBAL_COORDS *= local.inv();
		output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

		cout << k << endl;
	}
}

// эта одометрия работает с оптимизацией без регуляризаторов и скоростей
/*void optimized_odometry(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file)
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1);
	SimpleStruct frame;
	for (int k = 0; k < number_of_iterations; k += step)
	{

		vector<Point> ground_points;
		vector<Point> ground_points_next;
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		KeyPointMatches kpm = align_images(frame.current, frame.next, 1000);
		vector<Point2f> kp1, kp2;
		for (auto& match : kpm.matches)
		{
			float u = (float(kpm.kp1.at(match.queryIdx).pt.x));
			float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
			if (dynamic_classes.size() != 0)
				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(v), int(u)))) != dynamic_classes.end())
					continue;

			if (!is_dynamic(frame.current, frame.mask, Point(u, v), dynamic_classes)) // проверка окрестности точки на наличие динамических классов 
			{
				kp1.push_back({ u, v });
				kp2.push_back({ float(kpm.kp2.at(match.trainIdx).pt.x) , float(kpm.kp2.at(match.trainIdx).pt.y) });
				if (int(frame.mask.at<uchar>(int(v), int(u))) == 0) // если точка лежит на земле и в окрестности нету динамических точек
				{
					ground_points.emplace_back(Point(u, v));
					ground_points_next.emplace_back(Point(float(kpm.kp2.at(match.trainIdx).pt.x), float(kpm.kp2.at(match.trainIdx).pt.y))); // эти точки то надо для оптимизации, а не верхние 
				}
			}
		}
		Mat E, R, t, useless_masks;
		E = findEssentialMat(kp1, kp2, camera.K, RANSAC, 0.99, 1.0, useless_masks);
		recoverPose(E, kp1, kp2, camera.K, R, t, useless_masks);

		t.convertTo(t, CV_32FC1);
		R.convertTo(R, CV_32FC1);

		t *= (0.001 * frame.dt * frame.speed);

		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));
		// на данный момент имеем оценку вектора смещения и матрицу поворота - оптимизируем их 

		// составляем массив double* для Ceres из оценок точек в R3 в с.к. камеры

		vector<double*> ground_points_3d_for_optimized; // точки для оптимизации 
		for (auto& pt : ground_points)
		{
			Mat pt3d = estimate(pt, camera.R0, camera.t0, camera.K);
			double* buffer = new double[3]; // выделяем память
			buffer[0] = pt3d.at<float>(0, 0);
			buffer[1] = pt3d.at<float>(0, 1);
			buffer[2] = pt3d.at<float>(0, 2);
			ground_points_3d_for_optimized.emplace_back(buffer);
		}

		vector<double> rt = get_angles_and_vec(local); // выделили вектор {угол; угол; угол; tx; ty; tz}
		double angles_and_vecs_for_optimize[] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };
		//////////////////////////////			ОПТИМИЗАЦИЯ				////////////////////////////////
		// формируем проблему
		ceres::Problem problem;
		for (int k = 0; k < ground_points_3d_for_optimized.size(); ++k)
		{
			ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(ground_points_next[k].x), double(ground_points_next[k].y),
				camera.K.at<float>(0, 0), camera.K.at<float>(1, 1), camera.K.at<float>(0, 2), camera.K.at<float>(1, 2));
			problem.AddResidualBlock(cost_function, nullptr, angles_and_vecs_for_optimize, ground_points_3d_for_optimized[k]);

			// фиксируем изначальные значения 
			double initial_point_movement[6];
			double initial_point_pt[3];
			for (int m = 0; m < 3; ++m)
				initial_point_pt[m] = ground_points_3d_for_optimized[k][m];
			for (int m = 0; m < 6; ++m)
				initial_point_movement[m] = angles_and_vecs_for_optimize[m];
			// ограничения на углы и вектор смещения 
			for (int idx = 0; idx < 6; ++idx)
			{
				if (initial_point_movement[idx] >= 0)
				{
					problem.SetParameterLowerBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 0.80);
					problem.SetParameterUpperBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 1.20);
				}
				else
				{
					problem.SetParameterLowerBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 1.20);
					problem.SetParameterUpperBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 0.80);
				}
			}
			// ограничения на координаты
			for (int idx = 0; idx < 3; ++idx)
			{
				if (initial_point_pt[idx] >= 0)
				{
					problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
					problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
				}
				else
				{
					problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
					problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
				}
			}
		}
		// решаем проблему
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		///////////////////////////			ОПТИМИЗАЦИЯ				///////////////////////////////////

		////////////////////////// ПЕРЕХОД В ГЛОБАЛЬНУЮ СК          //////////////////////////////////

		local = reconstruct_from_v6(angles_and_vecs_for_optimize);

		local(Rect(0, 0, 3, 3)).copyTo(R);
		local(Rect(3, 0, 1, 3)).copyTo(t);

		R = camera.R0.inv() * R * camera.R0;
		t = camera.R0.inv() * t;

		local = Mat::eye(Size(4, 4), CV_32FC1);

		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		////////////////////////// ПЕРЕХОД В ГЛОБАЛЬНУЮ СК          //////////////////////////////////

		GLOBAL_COORDS *= local.inv();
		output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

		for (auto& pt : ground_points_3d_for_optimized)
			delete[] pt;
		cout << k << endl;
	}
}*/


// это финальная одометрия, которая работает с регуляризаторами и самостоятельной оценкой скорости 
void autonomous_odometry(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file)
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1);
	SimpleStruct frame;

	Mat plane(Size(4, 1), CV_32FC1); // z = 0;
	plane.at<float>(0, 0) = 0;
	plane.at<float>(0, 1) = 0;
	plane.at<float>(0, 2) = 1;
	plane.at<float>(0, 3) = 0;

	for (int k = 0; k < number_of_iterations; k += step)
	{

		vector<Vec3f> ground_points_3d_camera;
		vector<Vec3f> ground_points_3d_world;
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		KeyPointMatches kpm = align_images(frame.current, frame.next, 1000);
		vector<Point2f> image_points;
		vector<Point2f> start_points, end_points;
		for (auto& match : kpm.matches)
		{
			float u = (float(kpm.kp1.at(match.queryIdx).pt.x));
			float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
			if (dynamic_classes.size() != 0)
				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(v), int(u)))) != dynamic_classes.end())
					continue;

			if (!is_dynamic(frame.current, frame.mask, Point(u, v), dynamic_classes)) // проверка окрестности точки на наличие динамических классов 
			{
				if (int(frame.mask.at<uchar>(int(v), int(u))) == 0) // если точка лежит на земле и в окрестности нету динамических точек
				{
					Mat pt_world = general_estimate(Point(int(u), int(v)), camera.R0, camera.t0, camera.K, plane); // pt3d уже для перевода в R3 в world

					if (pt_world.at<float>(0, 0) > -5 && pt_world.at<float>(0, 0) < 5)
					{
						start_points.emplace_back(Point2f(u, v));
					}
				}
			}
		}
		Mat current_gray, next_gray;
		cvtColor(frame.current, current_gray, COLOR_BGR2GRAY);
		cvtColor(frame.next, next_gray, COLOR_BGR2GRAY);
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(current_gray, next_gray, start_points, end_points, status, err, Size(15, 15), 2, criteria);
		vector<double*> ground_points_3d_for_optimized; // точки для оптимизации 
		for (uint i = 0; i < start_points.size(); i++)
		{
			if (status[i] == 1)
			{
				double* buffer = new double[3]; // выделяем память
				Mat pt_world = general_estimate(start_points[i], camera.R0, camera.t0, camera.K, plane);
				Mat pt = estimate(start_points[i], camera.R0, camera.t0, camera.K);
				ground_points_3d_world.emplace_back(Vec3f(pt_world.at<float>(0, 0), pt_world.at<float>(1, 0), pt_world.at<float>(2, 0)));
				buffer[0] = pt_world.at<float>(0, 0);
				buffer[1] = pt_world.at<float>(1, 0);
				buffer[2] = pt_world.at<float>(2, 0);
				ground_points_3d_camera.emplace_back(Vec3f(pt.at<float>(0, 0), pt.at<float>(1, 0), pt.at<float>(2, 0)));
				image_points.push_back(end_points[i]);
				ground_points_3d_for_optimized.emplace_back(buffer);
			}
		}
		Mat rvec, tvec;
		solvePnPRansac(ground_points_3d_camera, image_points, camera.K, noArray(), rvec, tvec);
		Mat R, t(tvec); // проверь, что конструктор нормально работает
		Rodrigues(rvec, R);
		t.convertTo(t, CV_32FC1);
		R.convertTo(R, CV_32FC1);

		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));
		cout << t << endl;
		// на данный момент имеем оценку вектора смещения и матрицу поворота - оптимизируем их 

		// составляем массив double* для Ceres из оценок точек в R3 в с.к. камеры

		vector<double> rt = get_angles_and_vec(local); // выделили вектор {угол; угол; угол; tx; ty; tz}
		double angles_and_vecs_for_optimize[] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };
		//////////////////////////////			ОПТИМИЗАЦИЯ				////////////////////////////////
		// формируем проблему
		ceres::Problem problem;

		/// ПРОВЕРКА ФОРМУЛ ///
		/*for (int c = 0; c < ground_points_3d_camera.size(); ++c)
		{
			//cout << R << endl << camera.R0.inv()*t << endl;
			Mat point(Size(1, 4), CV_32FC1);
			point.at<float>(0, 0) = ground_points_3d_world[c][0];
			point.at<float>(1, 0) = ground_points_3d_world[c][1];
			point.at<float>(2, 0) = ground_points_3d_world[c][2];
			point.at<float>(3, 0) = 1;

			Mat Rt = Mat::eye(Size(4, 3), CV_32FC1);

			camera.R0.copyTo(Rt(Rect(0, 0, 3, 3)));
			camera.t0.copyTo(Rt(Rect(3, 0, 1, 3)));

			Mat answer = camera.K*(R*(Rt*point)+t);
			answer /= answer.at<float>(2, 0);
			cout << answer.at<float>(0, 0) << " " << answer.at<float>(1, 0) << " " << image_points[c].x << " " << image_points[c].y << endl;
		}*/
		/// ПРОВЕРКА ФОРМУЛ ///
		for (int k = 0; k < ground_points_3d_for_optimized.size(); ++k)
		{
			ceres::CostFunction* cost_function = SnavelyReprojectionErrorWorld::Create(double(image_points[k].x), double(image_points[k].y),
				camera.K.at<float>(0, 0), camera.K.at<float>(1, 1), camera.K.at<float>(0, 2), camera.K.at<float>(1, 2), camera.R0, camera.t0);
			problem.AddResidualBlock(cost_function, nullptr, angles_and_vecs_for_optimize, ground_points_3d_for_optimized[k]);


			double start[6];
			for (int p = 0; p < 6; ++p)
				start[p] = angles_and_vecs_for_optimize[p];

			for (int p = 0; p < 6; ++p) // ограниения на изменение значения - +- 20% по координате и углам
			{
				if (start[p] >= 0)
				{
					problem.SetParameterLowerBound(angles_and_vecs_for_optimize, p, start[p] * 0.80);
					problem.SetParameterUpperBound(angles_and_vecs_for_optimize, p, start[p] * 1.20);
				}
				else
				{
					problem.SetParameterLowerBound(angles_and_vecs_for_optimize, p, start[p] * 1.20);
					problem.SetParameterUpperBound(angles_and_vecs_for_optimize, p, start[p] * 0.80);
				}
			}
			double initial_point_pt[3];
			for (int m = 0; m < 3; ++m)
				initial_point_pt[m] = ground_points_3d_for_optimized[k][m];
			for (int idx = 0; idx < 3; ++idx)
			{
				if (ground_points_3d_for_optimized[k][idx] >= 0)
				{
					problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
					problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
				}
				else
				{
					problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
					problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
				}
			}
			problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], 2, 0);
			problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], 2, 0.1);
		}
		// решаем проблему
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		///////////////////////////			ОПТИМИЗАЦИЯ				///////////////////////////////////

		////////////////////////// ПЕРЕХОД В ГЛОБАЛЬНУЮ СК          //////////////////////////////////

		local = reconstruct_from_v6(angles_and_vecs_for_optimize);

		local(Rect(0, 0, 3, 3)).copyTo(R);
		local(Rect(3, 0, 1, 3)).copyTo(t);

		R = camera.R0.inv() * R * camera.R0;
		t = camera.R0.inv() * t;

		local = Mat::eye(Size(4, 4), CV_32FC1);

		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		////////////////////////// ПЕРЕХОД В ГЛОБАЛЬНУЮ СК          //////////////////////////////////

		GLOBAL_COORDS *= local.inv();
		output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

		for (auto& pt : ground_points_3d_for_optimized)
			delete[] pt;
		cout << k << endl;
	}
}
