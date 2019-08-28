#pragma once

#ifndef __RECOGNITION__
#define __RECOGNITION__

#include <opencv2/features2d/features2d.hpp>
#include "../Utils/CvUtils.hpp"
#include "MarkerlessDB.hpp"
#include "MarkerlessTrackable.hpp"
#include <map>

/**
 * @brief Class provides store and recognition marker image
 * 
 */
class Recognition {
private:
    MarkerlessDB *markerlessDb;                                // 'Bag of visual words' object
public:
    Recognition();

    /**
     * @brief Add image to marker database
     * 
     * @param img Marker image
     * @return int 
     */
    int addTrackImage(const cv::Mat &img);

    /**
     * @brief Make a query 
     * 
     * @param img Frame
     * @param amountRes Limit of results
     * @return std::vector<QueryItem> Query results
     */
    std::vector<QueryItem> queryImage(const cv::Mat &img, int amountRes = 1);

    ~Recognition();
};

#endif // __RECOGNITION__