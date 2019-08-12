#pragma once

#ifndef __RECOGNITION__
#define __RECOGNITION__

#include <opencv2/features2d/features2d.hpp>
#include "src/Utils/CvUtils.hpp"
#include "src/Recognition/BoVw.hpp"

//enum AlgorithmType{
//    SURF,
//    SIFT
//};

class Recognition {
private:
    cv::Ptr<cv::FeatureDetector> featureDetector;
    BoVW *vw;
public:
    Recognition();
    void createBagOfVisualWords(const std::vector<cv::Mat>& imgs, int numClusters = 0);
    void extractFeatures(const cv::Mat& img, std::vector<cv::KeyPoint> keyPoints, cv::Mat& descriptor);
    int addTrackImage(const cv::Mat& img, int id);
    int queryImage(const cv::Mat& img, int resId = 1);
    ~Recognition();

    int getFeatureIds(const cv::Mat& descriptor, std::vector<int>& ids);
    int storeImageFeatures(int id, const cv::Size& size, std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids);
    int
};

#endif // __RECOGNITION__