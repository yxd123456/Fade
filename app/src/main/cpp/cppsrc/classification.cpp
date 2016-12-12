#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
#include "classification.h"
using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
/*
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;*/
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {

  Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  void *ll;
  scale_ = 1;
  //std::shared_ptr<Net<float> > net_=(std::shared_ptr<Net<float> >);
  //std::shared_ptr<Net<float> > net_=dynamic_pointer_cast<Net<float>>(netFloat);
  std::shared_ptr<Net<float> > net_;
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  //std::cout<<mean_file<<std::endl;
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  labels_.clear();
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  netFloat=static_pointer_cast<void>(net_);
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
					   std::vector<int> mean_value,
                       const string& label_file) {

  Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  void *ll;
  scale_ = 1;
  //std::shared_ptr<Net<float> > net_=(std::shared_ptr<Net<float> >);
  //std::shared_ptr<Net<float> > net_=dynamic_pointer_cast<Net<float>>(netFloat);
  std::shared_ptr<Net<float> > net_;
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  //std::cout<<mean_file<<std::endl;
  SetMean(mean_value);


  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  labels_.clear();
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  netFloat=static_pointer_cast<void>(net_);
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

std::vector<std::vector<Prediction>> Classifier::ClassifyVecotrImage(std::vector<cv::Mat> imageList,int N)
{
    //std::vecotr<float> output=Predict(imageList);
    std::vector<float> output = PredictVector(imageList);
    std::vector<std::vector<Prediction>> predictionList;
    int num=output.size()/imageList.size();
    for (int i=0;i<imageList.size();i++)
    {
        const float *begin=&(output[num*(i)]);
        const float *end=&output[0]+num*(i+1);
        std::vector<float> outVec(begin,end);
        std::vector<int> maxN = Argmax(outVec, N);
        std::vector<Prediction> predictions;
        for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], outVec[idx]));
        }
        predictionList.push_back(predictions);
    }

  return predictionList;
}

void Classifier::SetMean(std::vector<int> values)
{
  int typeMat = CV_32FC3;
  if (num_channels_==1)
  {
      typeMat = CV_32FC1;
  }
  if (num_channels_==3)
  {
      typeMat = CV_32FC3;
  }
	std::vector<cv::Mat> channels;
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
			cv::Scalar(values[i]));
		channels.push_back(channel);
	}
	cv::merge(channels, mean_);
  
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::PredictVector(std::vector<cv::Mat> imageList)
{
  std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(imageList.size(), num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayerVector(&input_channels,imageList.size());

  PreprocessVector(imageList, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->num()*output_layer->channels();
  return std::vector<float>(begin, end);
}
std::vector<float> Classifier::Predict(const cv::Mat& img) {
  
    std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

std::vector<float> Classifier::extractFeatureFromNet(cv::Mat image,std::string layerName)
{
    //std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
    std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(image, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  /*Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
  */
  //std::cout<<layerName<<std::endl;
  const boost::shared_ptr<Blob<float> > feature_blob = net_
          ->blob_by_name(layerName);
  const float *feature_blob_data = feature_blob->cpu_data();
  //float *feature;
  std::vector<float> featureList;
  featureList.resize(feature_blob->offset(1));
  //memcpy(feature,feature_blob_data,feature_blob->offset(0));
  for (int i=0;i<feature_blob->offset(1);i++)
  {
      featureList[i]=feature_blob_data[i];
  }
  return featureList;

}
std::vector<float> Classifier::extractFeature(const cv::Mat &img,std::string layerName)
{
    std::vector<float> outFeature = extractFeatureFromNet(img,layerName);
    return outFeature;
}

std::vector<std::vector<float>> Classifier::extractFeatureList(std::vector<cv::Mat> imageList,std::string layerName)
{
    std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
    //std::cout<<imageList.size()<<std::endl;
    Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(imageList.size(), num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayerVector(&input_channels,imageList.size());

  PreprocessVector(imageList, &input_channels);

  net_->ForwardPrefilled();

  //std::cout<<layerName<<std::endl;
  const boost::shared_ptr<Blob<float> > feature_blob = net_
          ->blob_by_name(layerName);
  const float *feature_blob_data = feature_blob->cpu_data();
  //float *feature;
  
  std::vector<std::vector<float>> featureListOut;

  featureListOut.resize(imageList.size());
  

  for (int i=0;i<imageList.size();i++)
  {
  //memcpy(feature,feature_blob_data,feature_blob->offset(0));
    std::vector<float> featureList;
    featureList.resize(feature_blob->offset(1));
    for (int j=0;j<feature_blob->offset(1);j++)
    {
        featureList[j]=feature_blob_data[feature_blob->offset(i)+j];
    }
    featureListOut[i]=featureList;
  }
  return featureListOut;
}


void Classifier::WrapInputLayerVector(std::vector<cv::Mat>* input_channels,int numVector) {
    std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int j=0; j<numVector; j++)
  {
    for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
    }
  }
}
void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
       std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  //std::cout<<sample_float.type()<<std::endl;
  //std::cout<<sample_float.cols<<std::endl;
  //std::cout<<sample_float.rows<<std::endl;

  //std::cout<<mean_.type()<<std::endl;
  //std::cout<<mean_.cols<<std::endl;
  //std::cout<<mean_.rows<<std::endl;

  cv::subtract(sample_float, mean_, sample_normalized);
  sample_normalized = sample_normalized*scale_;
  //std::cout<<sample_normalized.type()<<std::endl;
  //std::cout<<sample_normalized.cols<<std::endl;
  //std::cout<<sample_normalized.rows<<std::endl;
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::PreprocessVector(std::vector<cv::Mat>& imgList,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
       std::shared_ptr<Net<float> > net_=static_pointer_cast<Net<float>>(netFloat);
  cv::Mat sample;
  cv::Mat img;
  for (int i=0;i<imgList.size();i++)
  {
      img=imgList[i];
      if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
      else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
      else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
      else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
      else
        sample = img;

      cv::Mat sample_resized;
      if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
      else
        sample_resized = sample;

      cv::Mat sample_float;
      if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
      else
        sample_resized.convertTo(sample_float, CV_32FC1);

      cv::Mat sample_normalized;
      cv::subtract(sample_float, mean_, sample_normalized);
	  //cv::Mat temp;
	  //cv::multiply(sample_normalized, temp, sample_normalized, scale_);
	  sample_normalized = sample_normalized*scale_;
      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      //cv::split(sample_normalized, *input_channels);
      cv::split(sample_normalized, &((*input_channels)[i*num_channels_]));
  }
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
            == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::setScale(float scale)
{
	scale_ = scale;
}
