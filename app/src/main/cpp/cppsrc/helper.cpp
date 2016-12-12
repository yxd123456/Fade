#include "helper.h"



cv::Rect perturb(cv::Rect facebox)
{
	float translation_x = (rand() % 20 - 10)*0.01;
	float translation_y = (rand() % 20 - 10)*0.01;
	float scaling = 1.0 + (rand() % 20 - 10)*0.015;
	//    cout << scaling << endl;
	auto tx_pixel = translation_x * facebox.width;
	auto ty_pixel = translation_y * facebox.height;
	// Because the reference point is on the top left and not in the center, we
	// need to temporarily store the new width and calculate half of the offset.
	// We need to move it further to compensate for the scaling, i.e. keep the center the center.
	auto perturbed_width = facebox.width * scaling;
	auto perturbed_height = facebox.height * scaling;
	//auto perturbed_width_diff = facebox.width - perturbed_width;
	//auto perturbed_height_diff = facebox.height - perturbed_height;
	// Note: Rounding?
	cv::Rect perturbed_box(facebox.x + (facebox.width - perturbed_width) / 2.0f + tx_pixel, facebox.y + (facebox.height - perturbed_height) / 2.0f + ty_pixel, perturbed_width, perturbed_height);

	return perturbed_box;
}


cv::Mat align_mean(cv::Mat mean, cv::Rect facebox, float scaling_x, float scaling_y, float translation_x, float translation_y)
{
	using cv::Mat;
	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat aligned_mean = mean.clone();
	Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2);
	Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols);
	aligned_mean_x = (aligned_mean_x*scaling_x + 0.5f + translation_x) * facebox.width + facebox.x;
	aligned_mean_y = (aligned_mean_y*scaling_y + 0.3f + translation_y) * facebox.height + facebox.y;
	return aligned_mean;
}


cv::Mat align_mean(cv::Mat mean, cv::Mat landmarks)
{
	using cv::Mat;

	static float scaling_x = 1.0f;
	static float scaling_y = 1.0f;
	static float translation_x = 0.0f;
	static float translation_y = 0.0f;
	static bool isFirstCalled = true;
	if (isFirstCalled){
		isFirstCalled = false;
		auto   num_landmarks = mean.cols / 2;
		double min_x_val, max_x_val, min_y_val, max_y_val;
		cv::minMaxLoc(mean.colRange(0, num_landmarks), &min_x_val, &max_x_val);
		cv::minMaxLoc(mean.colRange(num_landmarks, mean.cols), &min_y_val, &max_y_val);
		scaling_x = 1.0f / (max_x_val - min_x_val);
		scaling_y = 1.0f / (max_y_val - min_y_val);
		translation_x = 0.0f;
		translation_y = 0.0f;
	}

	static auto num_landmarks = landmarks.cols / 2;
	double min_x_val, max_x_val, min_y_val, max_y_val;
	cv::minMaxLoc(landmarks.colRange(0, num_landmarks), &min_x_val, &max_x_val);
	cv::minMaxLoc(landmarks.colRange(num_landmarks, landmarks.cols), &min_y_val, &max_y_val);
	double width = max_x_val - min_x_val;
	double height = max_y_val - min_y_val;

	// Initial estimate x_0: Center the mean face at the [-0.5, 0.5] x [-0.5, 0.5] square (assuming the face-box is that square)
	// More precise: Take the mean as it is (assume it is in a space [-0.5, 0.5] x [-0.5, 0.5]), and just place it in the face-box as
	// if the box is [-0.5, 0.5] x [-0.5, 0.5]. (i.e. the mean coordinates get upscaled)
	Mat aligned_mean = mean.clone();
	Mat aligned_mean_x = aligned_mean.colRange(0, aligned_mean.cols / 2);
	Mat aligned_mean_y = aligned_mean.colRange(aligned_mean.cols / 2, aligned_mean.cols);
	aligned_mean_x = (aligned_mean_x*scaling_x + 0.5f + translation_x) * width + min_x_val;
	aligned_mean_y = (aligned_mean_y*scaling_y + 0.3f + translation_y) * height + min_y_val;
	return aligned_mean;
}

std::string trim(const std::string& str)
{
	std::string::size_type pos = str.find_first_not_of(' ');
	if (pos == std::string::npos)
	{
		return str;
	}
	std::string::size_type pos2 = str.find_last_not_of(' ');
	if (pos2 != std::string::npos)
	{
		return str.substr(pos, pos2 - pos + 1);
	}
	return str.substr(pos);
}

std::string replace(const std::string& str, const std::string& dest, const std::string& src)
{
	std::string ret = str;
	size_t pos = ret.find(dest);
	while (pos != std::string::npos){
		ret = ret.replace(pos, dest.length(), src);
		pos = ret.find(dest);
	}
	return ret;
}

std::vector<std::string> split(const  std::string& s, const std::string& delim)
{
	std::vector<std::string> elems;
	size_t pos = 0;
	size_t len = s.length();
	size_t delim_len = delim.length();
	if (delim_len == 0) return elems;
	while (pos < len)
	{
		int find_pos = s.find(delim, pos);
		if (find_pos < 0)
		{
			elems.push_back(s.substr(pos, len - pos));
			break;
		}
		elems.push_back(s.substr(pos, find_pos - pos));
		pos = find_pos + delim_len;
	}
	return elems;
}