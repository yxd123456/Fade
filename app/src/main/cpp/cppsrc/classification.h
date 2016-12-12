#ifndef __CLASSIFICATION_H__
#define __CLASSIFICATION_H__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <memory>
#include <string>
using std::string;
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);
  /*
  Classifier(const string& model_file,
             const string& trained_file,
             const string& label_file);*/
  Classifier(const string& model_file,
			 const string& trained_file,
			 std::vector<int> mean_value,
			 const string& label_file);
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);
  std::vector<std::vector<Prediction>> ClassifyVecotrImage(std::vector<cv::Mat> imageList,int N=1);

  std::vector<float> extractFeature(const cv::Mat &img,std::string layerName);
  std::vector<std::vector<float>> extractFeatureList(std::vector<cv::Mat> imageList,std::string layerName);
  std::vector<float> extractFeatureFromNet(cv::Mat image,std::string layerName);
  void setScale(float scale);
 private:
  void SetMean(const string& mean_file);
  //void SetMean();
  void SetMean(std::vector<int> mean_value);
  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void WrapInputLayerVector(std::vector<cv::Mat>* input_channels,int numVecotr);
  std::vector<float> PredictVector(std::vector<cv::Mat> imageList);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
  void PreprocessVector(std::vector<cv::Mat>& imgList,
                            std::vector<cv::Mat>* input_channels);
    std::shared_ptr<void > netFloat;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  float scale_;
  std::vector<string> labels_;
};

#endif // __CLASSIFICATION_H__