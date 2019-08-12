#pragma once
#ifndef AR_CORE_BOVW_HPP
#define AR_CORE_BOVW_HPP

#include <opencv2/flann/flann.hpp>
#include <opencv2/opencv.hpp>

struct QueryItem{
    int imgId;
    double probability;
    int amountMatched;
    std::vector<cv::Point2f> objPose;
    cv::Size imgSize;
    cv::Mat pose;
};

class BoVW {
private:
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
    float radius;
    int vote;
public:
    BoVW();
    void addFeatures(const cv::Mat& feature);

    void create(int amountCluster = 0);
    void clear();
    cv::Mat search(const cv::Mat &feature);

    int getVote(){return vote;}

    ~BoVW();

    std::vector<QueryItem>
    searchImageId(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids, cv::Size size, int amountRes);

    void voteQueryFeatures(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids);

    std::vector<QueryItem> getMatchResults(std::vector<cv::KeyPoint> keyPoints);

    std::vector<QueryItem>
    filterGeomResults(std::vector<cv::KeyPoint> keyPoints, std::vector<QueryItem> pre, cv::Size size, int amountRes);
};


#endif //AR_CORE_BOVW_HPP
