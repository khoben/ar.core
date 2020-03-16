#include "android.hpp"

std::shared_ptr<AR> ar;       // AR instance
const int maxFrameSize = 800; // maximum frame size

void init(Mat frame) {
    // init AR instance
    if (!ar)
        ar = std::make_shared<ARMarkerless>();
    ar->init(frame.size(), maxFrameSize);
}

int addMarker(cv::Mat img) {
    if (!ar)
        return -1;
    // load marker image
    ar->add(img);
    return 0;
}

int process(Mat frame, Mat &out) {
    if (!ar)
        return -1;
    std::vector<QueryItem> result = ar->process(frame);
    frame.copyTo(out);
    for (auto &r: result) {
        cv::Scalar val(255);
        Boundary objPose = r.objPose;
        cv::line(out, objPose[3], objPose[0], val, 3);
        for (int i = 0; i < 3; i++) {
            line(out, objPose[i], objPose[i + 1], val, 3);
        }
        cv::Point center((objPose[0] + objPose[2]) / 2);

        cv::putText(out, std::to_string(r.imgId), center, FONT_HERSHEY_TRIPLEX, 6, val, 3);
    }

    return 0;
}
