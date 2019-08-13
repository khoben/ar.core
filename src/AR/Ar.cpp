#include "Ar.hpp"

AR::~AR() = default;

AR::AR() {
    recognitionInstance = new Recognition();
    trackingInstance = Tracking::create();
}



int AR::process(cv::Mat frame) {
    auto result = recognitionInstance->queryImage(frame);
    if (!result.empty()){
        std::cout << "Match: img_id:" << result[0].imgId << std::endl;
    }else{
        std::cout << "No match" << std::endl;
    }
    return 0;
}

int AR::add(std::vector<cv::Mat> imgs) {
    recognitionInstance->createBagOfVisualWords(imgs);
    for(const auto& img: imgs) {
        recognitionInstance->addTrackImage(img);
    }
    return 0;
}

int AR::add(cv::Mat img) {
    return add(std::vector<cv::Mat>{img});
}

