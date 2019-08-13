#include "Ar.hpp"

AR::~AR() = default;

AR::AR() {
    recognitionInstance = new Recognition();
    trackingInstance = Tracking::create();
}



int AR::processFrame(cv::Mat frame) {
    imshow("edges", frame);
    return 0;
}

int AR::addToTrack(std::vector<cv::Mat> imgs) {
    return 0;
}

