#pragma once
#ifndef AR_CORE_BOVW_HPP
#define AR_CORE_BOVW_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

class BoVW {
private:
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
    float RADIUS;
    int VOTE_NUM;
public:
    BoVW();

    void addFeatures(const cv::Mat &feature);

    void create(int amountCluster = 0);

    void clear();

    cv::Mat search(const cv::Mat &feature);

    int getVote() { return VOTE_NUM; }

    void setVote(int vote) { VOTE_NUM = vote; }

    int size() const;

    ~BoVW();
};


#endif //AR_CORE_BOVW_HPP
