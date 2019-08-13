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
    int process(cv::Mat frame);
    int add(std::vector<cv::Mat> imgs);
    int add(cv::Mat img);
    ~AR();
};


#endif //AR_CORE_AR_HPP
