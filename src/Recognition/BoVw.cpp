#include "BoVw.hpp"

BoVW::BoVW() {
    descriptorMatcher = cv::DescriptorMatcher::create("FlannBased");
    VOTE_NUM = 1;
    RADIUS = 0.2;
}

void BoVW::addFeatures(const cv::Mat &feature) {
    descriptorMatcher->add(std::vector<cv::Mat>{feature});
}

void BoVW::clear() {
    descriptorMatcher->clear();
}

BoVW::~BoVW() {
    clear();
}

void BoVW::create(int amountCluster) {
    if (amountCluster > 0) {
        //TODO: k-means clustering
    } else {
        descriptorMatcher->train();
    }
}

cv::Mat BoVW::search(const cv::Mat &feature) {
    const int KNN_SIZE = VOTE_NUM + 1;
    int size = feature.size().height;

    cv::Mat id(size, KNN_SIZE, CV_32SC1);
    std::vector<std::vector<cv::DMatch>> matchId;
    descriptorMatcher->knnMatch(feature, matchId, KNN_SIZE);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < VOTE_NUM; ++j) {
            cv::DMatch match = matchId[i][j];
            if (match.distance >= RADIUS) {
                id.at<int>(i, j) = -1;
            } else {
                id.at<int>(i, j) = match.trainIdx;
            }
        }
    }

    return id;
}

