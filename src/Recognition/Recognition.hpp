#pragma once

#ifndef __RECOGNITION__
#define __RECOGNITION__

#include <opencv2/features2d/features2d.hpp>
#include "../Utils/CvUtils.hpp"
#include "FeatureDB.hpp"
#include <map>

/**
 * @brief Class provides store and recognition marker image
 * 
 */
class Recognition {
private:
    cv::Ptr<cv::FeatureDetector> featureDetector; // Feature detector
    FeatureDB *vw;                                // 'Bag of visual words' object
    int imageAmount;                              // Amount of marker images
    int featureAmount;                            // Amount of features
    int MIN_MATCH = 6;                            // Minimum number of required matches
    float MIN_PROBABILITY = 0.75f;                 // Minimum probability of successful match
    float DISTANTION_TOLERANCE = 5e-4;            // Distance tolerance between corners
    float MIN_PROBABILITY_SUCCESS_MATCH = MIN_PROBABILITY
            * MIN_PROBABILITY * MIN_PROBABILITY; // Minimal success match probability percentage

    std::unordered_multimap<int, FeatureInfo> featureStore;          // {Feature id: Feature} map
    std::unordered_map<int, cv::KeyPoint> keyPointStore;             // {KeyPoint id: KeyPoint} map
    std::unordered_map<int, ImageInfo> imageInfoStore;               // {Image id: ImageInfo} map
    std::unordered_map<int, std::vector<FeatureVote> *> voteStorage; // {Image id: (KeyPoint<->Feature) binding} map

public:
    Recognition();

    /**
     * @brief Add and create a BoVW
     * 
     * @param imgs Marker images
     * @param numClusters Num of clusters
     */
    void addAndCreateBagOfVisualWords(const std::vector<cv::Mat> &imgs, int numClusters = 0);

    /**
     * @brief Add image to BoVW
     * 
     * @param img Marker image
     */
    void addVisualWord(const cv::Mat &img);

    /**
     * @brief Create a Bag Of Visual Words 
     * 
     * @param numClusters amount of clusters
     */
    void createBagOfVisualWords(int numClusters = 0);

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

private:

    /**
     * @brief calculates the CDF
     *BIN(x, n, p) = n!/(x!*(n-x)!) p^x (1-p)^(n-x)
     *
     * @param x Random variable
     * @param n Total number of trials
     * @param p Probability of success of a single trial
     * @return float CDF value
     */
    float binomialCDF(int x, int n, float p);

    /**
     * @brief Extract features from image
     * 
     * @param img Image
     * @param keyPoints [out] KeyPoints
     * @param descriptor [out] Descriptors
     */
    void extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor);

    /**
     * @brief For descriptor get feature ids
     * 
     * @param descriptor Descriptor
     * @param ids [out] Feature ids
     * @return int 
     */
    int getFeatureIds(const cv::Mat &descriptor, std::vector<int> &ids);

    /**
     * @brief Store marker image features
     * 
     * @param id Id of marker
     * @param size Size of marker image
     * @param keyPoints Marker keypoints
     * @param ids Id of features
     * @return int 
     */
    int storeImageFeatures(int id, const cv::Size &size, std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids);

    /**
     * @brief Get the KeyPoint candidate
     * 
     * @return int Keypoint id
     */
    int getCandidateKpId();

    /**
     * @brief Make query
     * 
     * @param keyPoints KeyPoints of frame
     * @param ids  Ids of features frame image
     * @param size Frame size
     * @param amountWords Amount of visual words
     * @param amountRes Limit results
     * @return std::vector<QueryItem> Result of query
     */
    std::vector<QueryItem>
    searchImageId(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids, cv::Size size, int amountWords,
                  int amountRes);

    /**
     * @brief Find (image id <-> keypoint id <-> feature id) binds
     * 
     * @param keyPoints KeyPoints of frame
     * @param ids Ids of features
     */
    void voteQueryFeatures(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids);

    /**
     * @brief Clear (image id <-> keypoint id <-> feature id) binds
     * 
     */
    void clearVote();

    /**
     * @brief Get the Match Results
     * 
     * @param keyPoints KeyPoint
     * @param amountWords Amount of visual words
     * @return std::vector<QueryItem> Result of query
     */
    std::vector<QueryItem> getMatchResults(std::vector<cv::KeyPoint> keyPoints, int amountWords);

    /**
     * @brief Filter results
     * 
     * @param keyPoints KeyPoints
     * @param pre Unfiltered results
     * @param size  Frame size
     * @param amountRes Limit amount of results
     * @return std::vector<QueryItem> Query results
     */
    std::vector<QueryItem>
    filterGeomResults(std::vector<cv::KeyPoint> keyPoints, std::vector<QueryItem> pre, cv::Size size, int amountRes);

    /**
     * @brief Push query and result keypoints to `q` and `r` vectors
     * 
     * @param keyPoints Frame keypoints
     * @param voteTable Vote table
     * @param q [out] Query keypoints
     * @param r [out] Result keypoints
     */
    void
    findPointPair(std::vector<cv::KeyPoint> keyPoints, std::vector<FeatureVote> voteTable, std::vector<cv::Point2f> &q,
                  std::vector<cv::Point2f> &r);
};

#endif // __RECOGNITION__