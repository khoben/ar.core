#pragma once

#ifndef __TRACKING__
#define __TRACKING__

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <algorithm>
#include "src/Utils/CvUtils.hpp"

class Tracking
{
private:
    cv::Mat prevFrame;             // processed frame
    ObjectPosition objectPosition; // coords of object`s corners
    cv::Mat homography;            // homography
    int MIN_FEATURE_POINTS;          // min amount of founded feature points
    /* Corner detector props */
    int maxAmountCorners;           // maximum amount of corners
    double minQualityCorners;       // minimum quality of corners
    double minDistanceCorners;      // minimum distance of corners
    std::vector<cv::Point2f> corners; // corners
    /* ------------------- */

    /* calcOpticalFlowPyrLK props */
    std::vector<uchar> opticalFlowStatus; // status of tracking points
    /* ------------------- */
public:
    Tracking();

    /**
     * @brief Initializes new tracking instance
     * 
     * @return Tracking* - instance
     */
    static Tracking *create();

    /**
     * @brief Start tracking
     * 
     * @param frame - frame
     * @param pos - position
     */
    void start(const cv::Mat &frame, const ObjectPosition& pos);

    /**
     * @brief Trying continue tracking
     * 
     * @param frame - frame
     * @return true - tracking continues
     * @return false - tracking failes
     */
    bool keepTracking(const cv::Mat &frame);
    ~Tracking();
};

#endif // __TRACKING__