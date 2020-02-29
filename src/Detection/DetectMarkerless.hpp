#pragma once

#ifndef __RECOGNITION__
#define __RECOGNITION__

#include <opencv2/features2d/features2d.hpp>
#include <map>
#include "Detect.hpp"
#include "../Utils/CvUtils.hpp"
#include "../ARStore/MarkerlessStorage.hpp"
#include "../Tracking/MarkerlessTrackable.hpp"

/**
 * @brief Class provides recognition marker functionality
 * 
 */
class DetectMarkerless : public Detect {
public:
    DetectMarkerless();

    /**
     * @brief Add image to marker database
     * 
     * @param img Marker image
     * @return int 
     */
    int addTrackImage(const cv::Mat &img) override;

    /**
     * @brief Add descriptors to database
     * @param descriptors
     * @param keyPoints
     * @param size
     * @return
     */
    int addTrackImage(const cv::Mat &descriptors,
                      const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size);

    /**
     * @brief Make a query 
     * 
     * @param img Frame
     * @return std::vector<QueryItem> Query results
     */
    std::vector<QueryItem> queryImage(const cv::Mat &img) override;

    ~DetectMarkerless();
};

#endif // __RECOGNITION__