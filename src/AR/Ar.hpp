#pragma once

#ifndef AR_CORE_AR_HPP
#define AR_CORE_AR_HPP

#include "../Tracking/Tracking.hpp"
#include "../Recognition/Recognition.hpp"

/**
 * @brief AR class for handling tracking
 * and recognition functions
 * 
 */
class AR {
private:
    Tracking *trackingInstance;       // tracking instance
    Recognition *recognitionInstance; // recognition instance
public:
    AR();

    /**
     * @brief Start recognition&tracking process
     * 
     * @param frame Frame
     * @return std::vector<QueryItem> Result
     */
    std::vector<QueryItem> process(const cv::Mat &frame);

    /**
     * @brief Start tracking process on detected marker image
     * 
     * @param frame Frame
     * @param pose Object pose
     */
    void startTracking(const cv::Mat &frame, const ObjectPosition &pose);

    /**
     * @brief Continue tracking
     * 
     * @param frame Frame
     * @return true Keep tracking
     * @return false Detected object lost
     */
    bool keepTracking(const cv::Mat &frame);

    /**
     * @brief Get the Tracking Instance object
     * 
     * @return Tracking* 
     */
    Tracking *getTrackingInstance() { return trackingInstance; }

    /**
     * @brief Add marker image objects
     * 
     * @param imgs marker images
     * @return int 
     */
    int addAll(const std::vector<cv::Mat> &imgs);

    /**
     * @brief Add marker image
     * 
     * @param img Marker image
     * @return int 
     */
    int add(const cv::Mat &img);

    int add(const cv::Mat &descriptors,
            const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size);

    ~AR();
};

#endif //AR_CORE_AR_HPP
