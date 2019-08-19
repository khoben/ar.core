#pragma once
#ifndef AR_CORE_FEATUREDB_HPP
#define AR_CORE_FEATUREDB_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "../Utils/CvUtils.hpp"

/**
 * @brief Class provides operations with features
 * 
 */
class FeatureDB {
private:
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher; // Descriptor matcher
    float RADIUS;                                     // Descriptor matcher nn ratio
    const int KNN_SIZE = 2;                           // knn size
public:
    FeatureDB();

    /**
     * @brief Add marker image feature
     * 
     * @param feature 
     */
    void addFeatures(const cv::Mat &feature);

    /**
     * @brief Create BoVW
     * 
     * @param amountCluster 
     */
    void create(int amountCluster = 0);

    /**
     * @brief Clear train descriptor collection
     * 
     */
    void clear();

    /**
     * @brief Search matches between features
     * 
     * @param feature Frame features
     * @return cv::Mat 
     */
    cv::Mat search(const cv::Mat &feature);

    /**
     * @brief Get amount of features
     * 
     * @return int 
     */
    int size() const;

    ~FeatureDB();
};

#endif //AR_CORE_FEATUREDB_HPP
