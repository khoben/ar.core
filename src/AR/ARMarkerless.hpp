#pragma once

#ifndef AR_CORE_ARMARKERLESS_HPP
#define AR_CORE_ARMARKERLESS_HPP

#include "AR.hpp"

/**
 * @brief AR class for handling tracking
 * and recognition functions
 * 
 */
class ARMarkerless : public AR {
public:
    ARMarkerless();

    /**
     * @brief Initialize AR instance: set querying image size
     *
     * @param frameSize source frame size
     * @param maxSize maximum size of querying image in pixels
     * for one of sides
     */
    void init(const cv::Size &frameSize, int maxSize) override;

    /**
     * @brief Start recognition&tracking process
     * 
     * @param frame Frame
     * @return std::vector<QueryItem> Result
     */
    std::vector<QueryItem> process(const cv::Mat &frame) override;

    /**
     * @brief Add marker image
     * 
     * @param img Marker image
     * @return int 
     */
    int add(const cv::Mat &img) override;

    /**
    * @brief Add marker image objects
    *
    * @param imgs marker images
    * @return int
    */
    int addAll(const std::vector<cv::Mat> &imgs) override;

    /**
     * @brief Add pre-calculated descriptors as marker
     *
     * @param descriptors
     * @param keyPoints
     * @param size
     * @return
     */
    int add(const cv::Mat &descriptors,
            const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size);

    ~ARMarkerless();
};

#endif //AR_CORE_ARMARKERLESS_HPP
