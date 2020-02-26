#pragma once
#ifndef AR_CORE_MARKERLESSDB_HPP
#define AR_CORE_MARKERLESSDB_HPP

#define _USE_MATH_DEFINES

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "../Utils/CvUtils.hpp"
#include "MarkerlessTrackable.hpp"

/**
 * @brief Class provides operations with features
 * 
 */
class MarkerlessDB {
private:
    cv::Ptr<cv::FeatureDetector> featureDetector;      // Feature detector
    std::vector<MarkerlessTrackable *> markers;
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher; // Descriptor matcher
    float RADIUS;                                     // Descriptor matcher nn ratio
    const int KNN_SIZE = 2;                           // knn size
    int markerAmount;                                  // Number of markers
    int featureAmount;                                  // Number of features
public:
    MarkerlessDB();

    /**
     * @brief Extract and add feature to database
     *
     * @param image
     */
    void add(const cv::Mat &image);

    /**
     * @brief Add feature to database
     *
     * @param descriptors
     * @param keyPoints
     * @param size
     */
    void add(const cv::Mat &descriptors,
             const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size);

    std::vector<QueryItem>
    match(const cv::Mat &queryImage, int minNumMatch = 10, float minProbability = 0.75f);

    /**
     * @brief Extract features from image
     *
     * @param img Image
     * @param keyPoints [out] KeyPoints
     * @param descriptor [out] Descriptors
     */
    void extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor);

    /**
     * @brief Add marker image feature
     * 
     * @param feature 
     */
    void addFeatures(const cv::Mat &feature);

    /**
     * @brief Get amount of features
     * 
     * @return int 
     */
    int size() const;

    ~MarkerlessDB();
};

#endif //AR_CORE_MARKERLESSDB_HPP
