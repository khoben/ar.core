#include "Ar.hpp"

AR::~AR() = default;

AR::AR() {
    recognitionInstance = new Recognition();
    trackingInstance = new Tracking();
}


std::vector<QueryItem> AR::process(const cv::Mat &frame) {
    return recognitionInstance->queryImage(frame, 10);

}

int AR::addAll(const std::vector<cv::Mat> &imgs) {
    for (const auto &img: imgs) {
        recognitionInstance->addTrackImage(img);
    }
    return 0;
}

int AR::add(const cv::Mat &img) {
    recognitionInstance->addTrackImage(img);
    return 0;
}


bool AR::keepTracking(const cv::Mat &frame) {
    return trackingInstance->keepTracking(frame);
}

void AR::startTracking(const cv::Mat &frame, const ObjectPosition &pose) {
    trackingInstance->start(frame, pose);
}

int AR::add(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keyPoints,
            const cv::Size &size) {
    recognitionInstance->addTrackImage(descriptors, keyPoints, size);
    return 0;
}



