#include "Ar.hpp"

AR::~AR() = default;

AR::AR() {
    recognitionInstance = new Recognition();
    trackingInstance = Tracking::create();
}



int AR::process(cv::Mat frame) {
    auto result = recognitionInstance->queryImage(frame, 10);
    if (!result.empty()){

        std::vector<cv::Point2f> objPose;
        for (auto r: result) {
            std::cout << "Match: img_id:" << r.imgId << std::endl;
            std::cout << "Pose" << r.objPose << r.probability << std::endl;
            objPose = r.objPose;
            cv::Scalar val(255);
            cv::line(frame, objPose[3], objPose[0], val);
            for (int i = 0; i < 3; i++) {
                line(frame, objPose[i], objPose[i + 1], val);
            }
        }
        namedWindow("result",cv::WINDOW_AUTOSIZE);
        imshow("result", frame);
        cv::waitKey(0);
    }else{
//        std::cout << "No match" << std::endl;
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

