#pragma once

#ifndef __CVUTILS__
#define __CVUTILS__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

// [0]:Top-Left
// [1]:Bottom-Left
// [2]:Bottom-Right
// [3]:Top-Right
typedef std::vector<cv::Point> ObjectPosition;

class CvUtils
{
public:
    /**
     * @brief Create a Mask object
     * 
     * @param size - size of frame
     * @param pose - pose of traking object
     * @return cv::Mat - Mask object
     */
    static cv::Mat createMask(cv::Size size, ObjectPosition pose)
    {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        cv::fillConvexPoly(mask, pose, cv::Scalar(255), 16, 0);
        return mask;
    }
};

#endif // __CVUTILS__