#include "opencv2/opencv.hpp"
#include "AR/Ar.hpp"
#include "Utils/CvUtils.hpp"

using namespace cv;

AR *ar;

const int maxFrameSize = 600;

cv::Mat makeQueryMat(cv::Size size, int max_size, int &scale) {
    int frame_max_size = std::max(size.width, size.height);
    scale = 1;
    while ((frame_max_size / scale) > max_size) {
        scale *= 2;
    }
    return cv::Mat(size.height / scale, size.width / scale, CV_8UC1);
}

int start() {
    cv::VideoCapture videoCapture;
    videoCapture.open(0);
    if (!videoCapture.isOpened()) {
        std::cout << "Failed to Open Camera" << std::endl;
        return -1;
    }
    cv::Mat frame, gray, query;
    videoCapture >> frame;
    int scale = 1;
    query = makeQueryMat(frame.size(), maxFrameSize, scale);

    bool isTracked = false;
    int imgId = -1;
    for (;;) {
        videoCapture >> frame;
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (isTracked) {
            std::cout << "Continue.. " << std::endl;
            isTracked = ar->keepTracking(gray);
        } else {
            imgId = -1;
            cv::resize(gray, query, query.size());
            std::vector<QueryItem> result = ar->process(query);
            if (!result.empty()) {
                std::vector<cv::Point2f> objPose;
                QueryItem r = result[0];

                std::cout << "Matched: img_id:" << r.imgId << std::endl;
                objPose = r.objPose;
                objPose = CvUtils::scalePoints(objPose, scale);
                std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
                ar->startTracking(gray, objPose);
                isTracked = true;
                imgId = r.imgId;
            }
        }
        if (isTracked&&imgId!=-1) {
            cv::Scalar val(255);
            ObjectPosition objPose = ar->getTrackingInstance()->objectPosition;
            cv::line(frame, objPose[3], objPose[0], val);
            for (int i = 0; i < 3; i++) {
                line(frame, objPose[i], objPose[i + 1], val);
            }

            cv::Point center((objPose[0] + objPose[2])/2);

            cv::putText(frame, std::to_string(imgId), center, FONT_HERSHEY_TRIPLEX, 6, val, 3);
        }
        cv::imshow("AR", frame);
        if (waitKey(1) == 27) break; //quit on ESC button
    }


    return 0;
}

int single(cv::Mat frame){
    cv::Mat gray, query;
    int scale = 1;
    query = makeQueryMat(frame.size(), maxFrameSize, scale);

    bool isTracked = false;
    int imgId = -1;

    if (frame.empty()) return -1;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, query, query.size());
    std::vector<QueryItem> result = ar->process(query);
    if (!result.empty()) {
        std::vector<cv::Point2f> objPose;
        QueryItem r = result[0];

        std::cout << "Matched: img_id:" << r.imgId << std::endl;
        objPose = r.objPose;
        objPose = CvUtils::scalePoints(objPose, scale);
        std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
        isTracked = true;
        imgId = r.imgId;

        cv::Scalar val(255);
        cv::line(frame, objPose[3], objPose[0], val);
        for (int i = 0; i < 3; i++) {
            line(frame, objPose[i], objPose[i + 1], val);
        }

        cv::Point center((objPose[0] + objPose[2])/2);

        cv::putText(frame, std::to_string(isTracked), center, FONT_HERSHEY_TRIPLEX, 6, val, 3);
    }

    cv::imshow("AR", frame);
    cv::waitKey();
    return 0;
}

int main(int, char **) {
    // init AR instance
    ar = new AR();
    // load marker images
    cv::Mat mat_1 = cv::imread(R"(..\resources\marker\miku.jpg)", 0);
    cv::Mat mat_3 = cv::imread(R"(..\resources\marker\314.png)", 0);
    cv::Mat mat_2 = cv::imread("..\\resources\\1.jpg");
//    ar->addAndCreate(std::vector<cv::Mat>{
//            mat_1,
//            mat
//    });
    ar->add(mat_1);
    ar->add(mat_3);
    ar->create();

    // start AR process
//    single(mat_2);
    start();

    return 0;
}