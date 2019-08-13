#pragma once

#ifndef __RECOGNITION__
#define __RECOGNITION__

#include <opencv2/features2d/features2d.hpp>
#include "src/Utils/CvUtils.hpp"
#include "src/Recognition/BoVw.hpp"
#include <map>

//enum AlgorithmType{
//    SURF,
//    SIFT
//};

struct QueryItem{
    int imgId;
    double probability;
    int amountMatched;
    std::vector<cv::Point2f> objPose;
    cv::Size imgSize;
    cv::Mat pose;
};

typedef struct{
    int in_feat_i;
    int keypoint_id;
}featureVote;

typedef struct{
    int keypoint_id;
    int img_id;
}featureInfo;

typedef struct{
    int feature_num;
    cv::Size img_size;
}imageInfo;

class Recognition {
private:
    cv::Ptr<cv::FeatureDetector> featureDetector;
    BoVW *vw;
    int imageAmount;
    int featureAmount;
    std::multimap<int, featureInfo> featureStore;
    std::map<int, cv::KeyPoint> keyPointStore;
    std::map<int, imageInfo> imageInfoStore;
    std::map<int, std::vector<featureVote>*> voteStorage;
public:
    Recognition();
    void createBagOfVisualWords(const std::vector<cv::Mat>& imgs, int numClusters = 0);
    void extractFeatures(const cv::Mat& img, std::vector<cv::KeyPoint> keyPoints, cv::Mat& descriptor);
    int addTrackImage(const cv::Mat& img, int id);
    std::vector<QueryItem> queryImage(const cv::Mat& img, int amountRes = 1);
    ~Recognition();

    int getFeatureIds(const cv::Mat& descriptor, std::vector<int>& ids);
    int storeImageFeatures(int id, const cv::Size& size, std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids);

    int getCandidateKpId();

    std::vector<QueryItem>
    searchImageId(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids, cv::Size size, int amountRes);

    void voteQueryFeatures(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids);

    std::vector<QueryItem> getMatchResults(std::vector<cv::KeyPoint> keyPoints);

    std::vector<QueryItem>
    filterGeomResults(std::vector<cv::KeyPoint> keyPoints, std::vector<QueryItem> pre, cv::Size size, int amountRes);

    void clearVote();
};

#endif // __RECOGNITION__