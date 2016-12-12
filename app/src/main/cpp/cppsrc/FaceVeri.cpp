#include<opencv2\opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\video\background_segm.hpp>
#include <math.h>
#include <iostream>
#include <numeric>
#include <vector>
#include"FaceVeri.h"
#include "classification.h"
#include "ldmarkmodel.h"

using namespace std;
using namespace cv;
using std::string;

#define MIN_FACE_SZ 24.0F

Classifier* classifier=NULL;


Mat cameraFrmR,IDCardFrmR;

float minFaceSzCamera, minFaceSzIDCard, scaleCamera, scaleIDCard, faceTolerance;
int cameraWidth, cameraHeight, cameraWidthR, cameraHeightR;
int IDCardWidth, IDCardHeight, IDCardWidthR, IDCardHeightR;

vector<float> featCamera, featIDCard;

Mat_<float> feature_a(1, FEAT_DIM), feature_b(1, FEAT_DIM);

float* fc=NULL, *fi=NULL;

Rect IDCardFaceArea;


Mat IDCardH(100, 256, CV_8UC1), CameraH(100, 256, CV_8UC1);

ldmarkmodel modelCamera,modelIDCard;

Mat rot_mat(2, 3, CV_64FC1);
const int crop_size = 128;
const int ec_mc_y = 48;
const int ec_y = 40;

Size csz, isz;
Mat img_crop(crop_size, crop_size, CV_8UC1);
Mat_<float> idcfts;


void bound(int x, int y, float ca, float sa, int *xmin, int *xmax, int *ymin, int *ymax)
/* int x,y;
float ca,sa;
int *xmin,*xmax,*ymin,*ymax;*/
{
	int rx, ry;
	// ?????
	rx = (int)floor(ca*(float)x + sa*(float)y);
	ry = (int)floor(-sa*(float)x + ca*(float)y);
	if (rx<*xmin) *xmin = rx; if (rx>*xmax) *xmax = rx;
	if (ry<*ymin) *ymin = ry; if (ry>*ymax) *ymax = ry;
}
void transformXY(float& x, float& y, float ang, Size s0, Size s1)
{
	float x0 = x - s0.width / 2.0f;
	float y0 = y - s0.height / 2.0f;
	x = x0*cos(ang) - y0*sin(ang) + s1.width / 2.0f;
	y = x0*sin(ang) + y0*cos(ang) + s1.height / 2.0f;
}
int guard(int x, int N)
{
	x = (x<0 ? 0 : x);
	x = (x>(N - 1) ? (N - 1) : x);
	return x;
}
void alignFace(Mat& img, Point2f le, Point2f re, Point2f mc)
{
	float angle = atan((le.y - re.y) / (le.x - re.x));

	int xmin, xmax, ymin, ymax, nx, ny;
	nx = img.cols; ny = img.rows;
	float ca, sa; ca = cos(angle); sa = sin(angle);
	xmin = xmax = ymin = ymax = 0;
	bound(0, 0, ca, sa, &xmin, &xmax, &ymin, &ymax);
	bound(nx - 1, 0, ca, sa, &xmin, &xmax, &ymin, &ymax);
	bound(0, ny - 1, ca, sa, &xmin, &xmax, &ymin, &ymax);
	bound(nx - 1, ny - 1, ca, sa, &xmin, &xmax, &ymin, &ymax);

	rot_mat.at<double>(0, 0) = ca;
	rot_mat.at<double>(0, 1) = sa;
	rot_mat.at<double>(0, 2) = -xmin;
	rot_mat.at<double>(1, 0) = -sa;
	rot_mat.at<double>(1, 1) = ca;
	rot_mat.at<double>(1, 2) = -ymin;
	int w = xmax - xmin + 1;
	int h = ymax - ymin + 1;
	Mat img_rot;
	warpAffine(img, img_rot, rot_mat, Size(w, h), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));
	//imshow("rot", img_rot);

	float x = (le.x + re.x) / 2.0f;
	float y = (le.y + re.y) / 2.0f;

	float ang = -angle;
	transformXY(x, y, ang, img.size(), img_rot.size());
	Point2f eyec(x, y);
	x = mc.x;
	y = mc.y;
	transformXY(x, y, ang, img.size(), img_rot.size());
	Point2f mouthc(x, y);

	float resize_scale = ec_mc_y / (mouthc.y - eyec.y);

	Mat img_resize;
	resize(img_rot, img_resize, Size(round(img_rot.cols*resize_scale), round(img_rot.rows*resize_scale)));

	Point eyec2(round(eyec.x*resize_scale), round(eyec.y*resize_scale));

	img_crop.setTo(255);
	int crop_y = eyec2.y - ec_y;
	int crop_y_end = crop_y + crop_size - 1;
	int crop_x = eyec2.x - floor(crop_size / 2.0);
	int crop_x_end = crop_x + crop_size - 1;

	Rect box(Point(guard(crop_x, img_resize.cols), guard(crop_y, img_resize.rows)), Point(guard(crop_x_end, img_resize.cols), guard(crop_y_end, img_resize.rows)));
	Rect box2 = box - Point(crop_x, crop_y);
	img_resize(box).copyTo(img_crop(box2));

	//imshow("crop", img_crop);
}
template <typename T>
std::vector<size_t> ordered(std::vector<T> const& values) {
	std::vector<size_t> indices(values.size());
	std::iota(begin(indices), end(indices), static_cast<size_t>(0));

	std::sort(
		begin(indices), end(indices),
		[&](size_t a, size_t b) { return values[a] < values[b]; }
	);
	return indices;
}
DLL_API void faceVeriInit(char* faceVeriConfigPath)
{
	FileStorage fs(faceVeriConfigPath, FileStorage::READ);

	string prototxtPath, caffemodelPath, labelPath, 
	faceDetectorPath, faceLandmarkDetectorPath, featDataPath;
	int isCPUMode;

	fs["minFaceSzCamera"] >> minFaceSzCamera;

	fs["minFaceSzIDCard"] >> minFaceSzIDCard;

	fs["cameraWidth"] >> cameraWidth;
	fs["cameraHeight"] >> cameraHeight;
	fs["IDCardWidth"] >> IDCardWidth;
	fs["IDCardHeight"] >> IDCardHeight;

	fs["prototxtPath"] >> prototxtPath;
	fs["caffemodelPath"] >> caffemodelPath;
	fs["labelPath"] >> labelPath;
	fs["faceDetectorPath"] >> faceDetectorPath;
	fs["faceLandmarkDetectorPath"] >> faceLandmarkDetectorPath;
	fs["featDataPath"] >> featDataPath;

	fs["isCPUMode"] >> isCPUMode;

	fs["IDCardFaceArea"] >> IDCardFaceArea;
	fs["faceTolerance"] >> faceTolerance;

	scaleCamera = minFaceSzCamera / MIN_FACE_SZ;
	scaleIDCard = minFaceSzIDCard / MIN_FACE_SZ;

	cameraWidthR = (int)round(cameraWidth / scaleCamera);
	cameraHeightR = (int)round(cameraHeight / scaleCamera);

	IDCardWidthR = (int)round(IDCardWidth / scaleIDCard);
	IDCardHeightR = (int)round(IDCardHeight / scaleIDCard);

	vector<int> vi;
	vi.push_back(0);
	classifier = new Classifier(prototxtPath, caffemodelPath, vi, labelPath);
	classifier->setScale(1.0 / 256.0);

	if (isCPUMode)
	{
		classifier->setMode(CPU_MODE);
	} 
	else
	{
		classifier->setMode(GPU_MODE);
		classifier->SetDevice(0);
	}

	csz = Size(minFaceSzCamera,minFaceSzCamera);
	isz = Size(minFaceSzIDCard,minFaceSzIDCard);
	fc = (float*)malloc(sizeof(float)* FEAT_DIM);
	fi = (float*)malloc(sizeof(float)* FEAT_DIM);


	Mat_<float> idcfts2;
	FileStorage fs1(featDataPath, FileStorage::READ);
	fs1["data"] >> idcfts2;
	fs1.release();
	idcfts.create(idcfts2.rows+1,idcfts2.cols);
	idcfts2.copyTo(idcfts(Rect(0, 1, idcfts2.cols, idcfts2.rows)));

	load_ldmarkmodel(faceLandmarkDetectorPath, modelCamera);
	load_ldmarkmodel(faceLandmarkDetectorPath, modelIDCard);
	modelCamera.loadFaceDetModelFile(faceDetectorPath);
	modelIDCard.loadFaceDetModelFile(faceDetectorPath);
	cout << "Initialization complete!" << endl;
}

DLL_API int faceFeatureExtractCamera(unsigned char* pFrame, FACERC& rc, float** featc, int isCompare)
{
	Mat originFrm(cameraHeight, cameraWidth, CV_8UC1, pFrame);
	cv::Mat current_shape;
	modelCamera.track(originFrm, current_shape, true, csz);
	cv::Vec3d eav;
	modelCamera.EstimateHeadPose(current_shape, eav);
	int numLandmarks = current_shape.cols / 2;
	if (numLandmarks && isCompare)
	{
		Point2f leye((current_shape.at<float>(36) + current_shape.at<float>(39)) / 2.0f, (current_shape.at<float>(36 + numLandmarks) + current_shape.at<float>(39 + numLandmarks)) / 2.0f);//
		Point2f reye((current_shape.at<float>(42) + current_shape.at<float>(45)) / 2.0f, (current_shape.at<float>(42 + numLandmarks) + current_shape.at<float>(45 + numLandmarks)) / 2.0f);//
		Point2f mouthc((current_shape.at<float>(48) + current_shape.at<float>(54)) / 2.0f, (current_shape.at<float>(48 + numLandmarks) + current_shape.at<float>(54 + numLandmarks)) / 2.0f);//
		float ct = (current_shape.at<float>(19 + numLandmarks) + current_shape.at<float>(24 + numLandmarks)) / 2.0f;
		Point tlp(current_shape.at<float>(1), ct), brp(current_shape.at<float>(15), current_shape.at<float>(8 + numLandmarks));
		Rect rc0(tlp, brp);
		alignFace(originFrm, leye, reye, mouthc);
		//nomalizeFace(img_crop);
		featCamera = classifier->extractFeature(img_crop, "eltwise_fc1");
		imshow("camera aaroi", img_crop);
		rc.x = rc0.x; 
		rc.y = rc0.y; 
		rc.width = rc0.width; 
		rc.height = rc0.height;
		//int h;
		//CameraH.setTo(0);
		for (size_t k = 0; k < FEAT_DIM; k++)
		{
			fc[k] = feature_a(0, k) = featCamera[k];
		}
		featc[0] = fc;
		return 1;
	}
	else if (!numLandmarks)
	{
		rc.x = 0;
		rc.y = 0;
		rc.width = 0;
		rc.height = 0;
		return 0;
	}
	else
	{
		Point2f leye((current_shape.at<float>(36) + current_shape.at<float>(39)) / 2.0f, (current_shape.at<float>(36 + numLandmarks) + current_shape.at<float>(39 + numLandmarks)) / 2.0f);//
		Point2f reye((current_shape.at<float>(42) + current_shape.at<float>(45)) / 2.0f, (current_shape.at<float>(42 + numLandmarks) + current_shape.at<float>(45 + numLandmarks)) / 2.0f);//
		Point2f mouthc((current_shape.at<float>(48) + current_shape.at<float>(54)) / 2.0f, (current_shape.at<float>(48 + numLandmarks) + current_shape.at<float>(54 + numLandmarks)) / 2.0f);//
		float ct = (current_shape.at<float>(19 + numLandmarks) + current_shape.at<float>(24 + numLandmarks)) / 2.0f;
		Point tlp(current_shape.at<float>(1), ct), brp(current_shape.at<float>(15), current_shape.at<float>(8 + numLandmarks));
		Rect rc0(tlp, brp);
		rc.x = rc0.x;
		rc.y = rc0.y;
		rc.width = rc0.width;
		rc.height = rc0.height;

		return 1;
	}
}

DLL_API int faceFeatureExtractIDCard(unsigned char* pFrame, FACERC& rc, float** feati)
{
	Mat originFrm(IDCardHeight, IDCardWidth, CV_8UC1, pFrame);
	cv::Mat current_shape;
	modelIDCard.track(originFrm, current_shape, true, isz);
	cv::Vec3d eav;
	modelIDCard.EstimateHeadPose(current_shape, eav);
	int numLandmarks = current_shape.cols / 2;
	if (numLandmarks)
	{
		Point2f leye((current_shape.at<float>(36) + current_shape.at<float>(39)) / 2.0f, (current_shape.at<float>(36 + numLandmarks) + current_shape.at<float>(39 + numLandmarks)) / 2.0f);//
		Point2f reye((current_shape.at<float>(42) + current_shape.at<float>(45)) / 2.0f, (current_shape.at<float>(42 + numLandmarks) + current_shape.at<float>(45 + numLandmarks)) / 2.0f);//
		Point2f mouthc((current_shape.at<float>(48) + current_shape.at<float>(54)) / 2.0f, (current_shape.at<float>(48 + numLandmarks) + current_shape.at<float>(54 + numLandmarks)) / 2.0f);//
		float ct = (current_shape.at<float>(19 + numLandmarks) + current_shape.at<float>(24 + numLandmarks)) / 2.0f;
		Point tlp(current_shape.at<float>(1), ct), brp(current_shape.at<float>(15), current_shape.at<float>(8 + numLandmarks));
		Rect rc0(tlp, brp);
		alignFace(originFrm, leye, reye, mouthc);
		//nomalizeFace(img_crop);
		featIDCard = classifier->extractFeature(img_crop, "eltwise_fc1");
		imshow("IDCard aaroi", img_crop);
		rc.x = rc0.x;
		rc.y = rc0.y;
		rc.width = rc0.width;
		rc.height = rc0.height;
		//int h;
		//IDCardH.setTo(0);
		for (size_t k = 0; k < FEAT_DIM; k++)
		{
			fi[k] = feature_b(0, k) = (float)featIDCard[k];
		}
		feati[0] = fi;
		return 2;
	}
	else
	{
		rc.x = IDCardFaceArea.x;
		rc.y = IDCardFaceArea.y;
		rc.width = IDCardFaceArea.width;
		rc.height = IDCardFaceArea.height;

		int hw = round((IDCardFaceArea.width + IDCardFaceArea.height)*1.5 / 2.0);
		cv::Rect rc1(IDCardFaceArea.x + IDCardFaceArea.width / 2 - hw / 2, IDCardFaceArea.y + IDCardFaceArea.height / 2 - round(hw*0.45), hw, hw);
		cv::Mat aaroi(hw, hw, CV_8UC1);
		aaroi.setTo(255);
		cv::Point tlp1(rc1.x<0 ? (-rc1.x) : 0, rc1.y<0 ? (-rc1.y) : 0),
			brp1(rc1.br().x>IDCardWidth ? IDCardWidth - rc1.x - 1 : hw - 1,
			rc1.br().y>IDCardHeight ? IDCardHeight - rc1.y - 1 : hw - 1);
		cv::Rect rc2(tlp1, brp1);//aaroi???
		cv::Point tlp3(rc1.x > 0 ? rc1.x : 0, rc1.y > 0 ? rc1.y : 0),
			brp3(rc1.br().x < IDCardWidth ? rc1.br().x - 1 : IDCardWidth - 1,
			rc1.br().y < IDCardHeight ? rc1.br().y - 1 : IDCardHeight - 1);
		cv::Rect rc3(tlp3, brp3);//origin 
		cv::Rect rc4(0, 0, IDCardWidth, IDCardHeight);
		originFrm(rc3).copyTo(aaroi(rc2));
		//imshow("aaroi", aaroi);

		//nomalizeFace(img_crop);
		featIDCard = classifier->extractFeature(aaroi, "eltwise_fc1");

		for (size_t k = 0; k < FEAT_DIM; k++)
		{
			fi[k] = feature_b(0, k) = (float)featIDCard[k];
		}
		feati[0] = fi;
		return 1;

	}
}

DLL_API int faceFeatureCompare(float* score)
{
	vector<float> mp(idcfts.rows); vector<size_t> idcs;
	for (int i = 0; i < FEAT_DIM;i++)
	{
		idcfts(0, i) = fi[i];
	}
	float aa = feature_a.dot(feature_a);
	for (int i = 0; i < idcfts.rows; i++)
	{
		float ac = feature_a.dot(idcfts.row(i));
		float cc = idcfts.row(i).dot(idcfts.row(i));
		mp[i] = ac / sqrt(aa*cc);
	}

	idcs = ordered(mp);
	int idx=INT_MIN;

	for (int i = 0; i < idcfts.rows; i++)
	{
		if (idcs[i]==0)
		{
			idx = i+1;
			break;
		}
	}
	cout << mp[idcs[idcfts.rows - 1]] << " -- " << mp[idcs[idx - 1]] << "   ";

	score[0] = mp[idcs[idx - 1]];

	score[1] = mp[idcs[idcfts.rows - 1]] - faceTolerance;

	if (score[0]>score[1])
	{
		return 1;
	} 
	else
	{
		return 0;
	}
}

DLL_API void faceVeriFree()
{
	if (classifier)
	{
		delete classifier;
	}
	if (fc)
	{
		free(fc);
	}
	if (fi)
	{
		free(fi);
	}
}