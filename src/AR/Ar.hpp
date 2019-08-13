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
    int processFrame(cv::Mat frame);
    int addToTrack(std::vector<cv::Mat> imgs);
    ~AR();
};


#endif //AR_CORE_AR_HPP
