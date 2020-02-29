#pragma once
#ifndef AR_CORE_MARKERLESSSTORAGE_HPP
#define AR_CORE_MARKERLESSSTORAGE_HPP

#define _USE_MATH_DEFINES

#include <cmath>
#include <opencv2/features2d/features2d.hpp>
#include "../Tracking/MarkerlessTrackable.hpp"
#include "../ARStore/Storage.hpp"

/**
 * @brief Class provides operations with features
 * 
 */
class MarkerlessStorage : public Storage {
private:
    // Markers
    std::vector<std::unique_ptr<MarkerlessTrackable>> markers;
    // Feature detector
    cv::Ptr<cv::FeatureDetector> featureDetector;
    // Descriptor matcher
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
    // Descriptor matcher nn ratio
    float RADIUS;
    // knn size
    const int KNN_SIZE = 2;
    // Min amount of matched features
    const int MIN_NUM_MATCH = 15;
    // Min value of probability
    const float MIN_PROBABILITY = 0.7f;
public:
    MarkerlessStorage();

    /**
     * @brief Extract and add feature to database
     *
     * @param image
     */
    void add(const cv::Mat &image) override;

    /**
     * @brief Add feature to database
     *
     * @param descriptors
     * @param keyPoints
     * @param size
     */
    void add(const cv::Mat &descriptors,
             const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size);

    /**
     * Try to find&match marker within storage
     *
     * @param queryImage Input image
     * @return Vector of results
     */
    std::vector<QueryItem> match(const cv::Mat &queryImage) override;

    /**
     * Try to find&match marker within storage
     *
     * @param queryImage Input image
     * @param minNumMatch Min amount of matched features
     * @param minProbability Min value of probability
     * @return Vector of results
     */
    std::vector<QueryItem>
    match(const cv::Mat &queryImage, int minNumMatch, float minProbability);

    /**
     * @brief Extract features from image
     *
     * @param img Image
     * @param keyPoints [out] KeyPoints
     * @param descriptor [out] Descriptors
     */
    void extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor);

    /**
     * @brief Add features to descriptor matcher
     * 
     * @param feature Features
     */
    void addFeatures(const cv::Mat &feature);

    /**
     * @brief Get amount of features
     * 
     * @return int 
     */
    [[nodiscard]] int size() const;

    ~MarkerlessStorage();
};

#endif //AR_CORE_MARKERLESSSTORAGE_HPP
