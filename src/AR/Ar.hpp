#pragma once

#ifndef AR_CORE_AR_HPP
#define AR_CORE_AR_HPP

#include "../Tracking/Tracking.hpp"
#include "../Recognition/Recognition.hpp"

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

    int addAndCreate(std::vector<cv::Mat> imgs);

    int add(cv::Mat img);
    int create();

    ~AR();
};


#endif //AR_CORE_AR_HPP
