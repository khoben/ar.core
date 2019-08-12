#include "BoVw.hpp"

BoVW::BoVW() {
    descriptorMatcher = cv::DescriptorMatcher::create("FlannBased");
    vote = 1;
    radius = 0.2;
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
    const int KNN_SIZE = vote + 1;
    int size = feature.size().height;

    cv::Mat id(size, KNN_SIZE, CV_32SC1);
    std::vector<std::vector<cv::DMatch>> matchId;
    descriptorMatcher->knnMatch(feature, matchId, KNN_SIZE);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < vote; ++j) {
            cv::DMatch match = matchId[i][j];
            if (match.distance >= radius) {
                id.at<int>(i, j) = -1;
            } else {
                id.at<int>(i, j) = match.trainIdx;
            }
        }
    }

    return id;
}

std::vector<QueryItem>
BoVW::searchImageId(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids, cv::Size size, int amountRes) {
    std::vector<QueryItem> queryReturn;
    voteQueryFeatures(keyPoints, ids);
    std::vector<QueryItem> tmp = getMatchResults(keyPoints);
    queryReturn = filterGeomResults(keyPoints, tmp, size, amountRes);
    return nullptr;
}

void BoVW::voteQueryFeatures(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids) {

}

std::vector<QueryItem> BoVW::getMatchResults(std::vector<cv::KeyPoint> keyPoints) {
    return std::vector<QueryItem>();
}

std::vector<QueryItem>
BoVW::filterGeomResults(std::vector<cv::KeyPoint> keyPoints, std::vector<QueryItem> pre, cv::Size size, int amountRes) {
    return std::vector<QueryItem>();
}

