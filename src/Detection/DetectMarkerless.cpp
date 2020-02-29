#include "DetectMarkerless.hpp"

DetectMarkerless::DetectMarkerless() {
    storage = std::make_shared<MarkerlessStorage>();
}

int DetectMarkerless::addTrackImage(const cv::Mat &img) {
    storage->add(img);
    return 0;
}

std::vector<QueryItem> DetectMarkerless::queryImage(const cv::Mat &img) {
    std::vector<QueryItem> queryReturn;
    queryReturn = storage->match(img);
    return queryReturn;
}


int
DetectMarkerless::addTrackImage(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keyPoints,
                                const cv::Size &size) {
    dynamic_cast<MarkerlessStorage &>(*storage).add(descriptors, keyPoints, size);
    return 0;
}

DetectMarkerless::~DetectMarkerless() = default;
