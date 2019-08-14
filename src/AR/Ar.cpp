#include "Ar.hpp"

AR::~AR() = default;

AR::AR() {
    recognitionInstance = new Recognition();
    trackingInstance = Tracking::create();
}


std::vector<QueryItem> AR::process(cv::Mat frame) {
    return recognitionInstance->queryImage(frame, 10);

}

int AR::add(std::vector<cv::Mat> imgs) {
    recognitionInstance->createBagOfVisualWords(imgs);
    for (const auto &img: imgs) {
        recognitionInstance->addTrackImage(img);
    }
    return 0;
}

int AR::add(cv::Mat img) {
    return add(std::vector<cv::Mat>{img});
}

bool AR::keepTracking(const cv::Mat &frame) {
    return trackingInstance->keepTracking(frame);
}

bool AR::startTracking(const cv::Mat &frame, const ObjectPosition &pose) {
    trackingInstance->start(frame, pose);
    return true;
}


