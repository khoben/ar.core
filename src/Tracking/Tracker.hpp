#pragma once

#ifndef __TRACKING__
#define __TRACKING__

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <algorithm>
#include "../Utils/CvUtils.hpp"

/**
 * @brief Class provides tracking
 */
class Tracker {
public:
    // coords of object`s corners
    Boundary objectPosition;
    // corners
    std::vector<cv::Point2f> corners;
private:
    // processed frame
    cv::Mat prevFrame;
    // homography
    cv::Mat homography;
    // min amount of founded feature points
    int MIN_FEATURE_POINTS;
    /* Corner detector props */
    // maximum amount of corners
    int maxAmountCorners;
    // minimum quality of corners
    double minQualityCorners;
    // minimum distance of corners
    double minDistanceCorners;
    /* ------------------- */

    /* calcOpticalFlowPyrLK props */
    // status of tracking points
    std::vector<uchar> opticalFlowStatus;
    std::vector<cv::Mat> prevPyr, nextPyr;
    /* ------------------- */
public:
    Tracker();

    /**
     * @brief Start tracking
     * 
     * @param frame - frame
     * @param pos - position
     */
    void start(const cv::Mat &frame, const Boundary &pos);

    /**
     * @brief Trying continue tracking
     * 
     * @param frame - frame
     * @return true - tracking continues
     * @return false - tracking failes
     */
    bool keepTracking(const cv::Mat &frame);

    /**
     * @brief Get the Homography matrix
     * 
     * @return cv::Mat Homography matrix
     */
    cv::Mat getHomography() { return homography; }

    ~Tracker();
};

#endif // __TRACKING__