#include "Ar.hpp"

AR::~AR() = default;

AR::AR() {
    recognitionInstance = new Recognition();
    trackingInstance = Tracking::create();
}


std::vector<QueryItem> AR::process(cv::Mat frame) {
    return recognitionInstance->queryImage(frame, 10);

}

int AR::addAndCreate(std::vector<cv::Mat> imgs) {
    recognitionInstance->addAndCreateBagOfVisualWords(imgs);
    for (const auto &img: imgs) {
        recognitionInstance->addTrackImage(img);
    }
    return 0;
}

int AR::add(cv::Mat img) {
    recognitionInstance->addVisualWord(img);
    recognitionInstance->addTrackImage(img);
    return 0;
}

int AR::create() {
    recognitionInstance->createBagOfVisualWords();
    return 0;
}

bool AR::keepTracking(const cv::Mat &frame) {
    return trackingInstance->keepTracking(frame);
}

void AR::startTracking(const cv::Mat &frame, const ObjectPosition &pose) {
    trackingInstance->start(frame, pose);
}



