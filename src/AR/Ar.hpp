#pragma once

#ifndef AR_CORE_AR_HPP
#define AR_CORE_AR_HPP

#include "src/Tracking/Tracking.hpp"
#include "src/Recognition/Recognition.hpp"

class AR {
private:
    Tracking *trackingInstance;
    Recognition *recognitionInstance;
public:
    AR();

    std::vector<QueryItem> process(cv::Mat frame);

    bool startTracking(const cv::Mat &frame, const ObjectPosition &pose);

    bool keepTracking(const cv::Mat &frame);

    Tracking *getTrackingInstance() { return trackingInstance; }

    int add(std::vector<cv::Mat> imgs);

    int add(cv::Mat img);

    ~AR();
};


#endif //AR_CORE_AR_HPP
