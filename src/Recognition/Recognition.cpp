#include "Recognition.hpp"

Recognition::Recognition() {
    //TODO: init detector and descriptor
    markerlessDb = new MarkerlessDB();
}

int Recognition::addTrackImage(const cv::Mat &img) {
    markerlessDb->add(img);
    return 0;
}

std::vector<QueryItem> Recognition::queryImage(const cv::Mat &img, int amountRes) {
    std::vector<QueryItem> queryReturn;
    queryReturn = markerlessDb->match(img);
    return queryReturn;
}

Recognition::~Recognition() = default;

int
Recognition::addTrackImage(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keyPoints,
                           const cv::Size &size) {

    markerlessDb->add(descriptors, keyPoints, size);
    return 0;
}
