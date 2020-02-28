#ifndef AR_CORE_STORAGE_HPP
#define AR_CORE_STORAGE_HPP

#include "../Utils/CvUtils.hpp"
#include <opencv2/opencv.hpp>
#include "../Tracking/Trackable.hpp"

/**
 * Base class for marker storage
 */
class Storage {
protected:
    // Number of markers
    int markerAmount{};
public:
    /**
     * @brief Add trackable image to database
     *
     * @param image
     */
    virtual void add(const cv::Mat &image) = 0;

    /**
     * Try to find&match marker within storage
     *
     * @param queryImage Input image
     * @return vector of results
     */
    virtual std::vector<QueryItem>
    match(const cv::Mat &queryImage) = 0;
};


#endif //AR_CORE_STORAGE_HPP
