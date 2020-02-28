#include "opencv2/opencv.hpp"
#include "AR/ARMarkerless.hpp"
#include <climits>
#include <chrono>

using namespace cv;

const int OPENCV_FONT = FONT_HERSHEY_DUPLEX; // OpenCV font

/**
 * @brief Find the object on videostream
 *
 * @param ar AR instance
 * @return int 
 */
int start(AR *ar);

/**
 * @brief Find object on single frame
 *
 * @param ar AR instance
 * @param frame 
 * @return int 
 */
int single(AR *ar, cv::Mat frame);

/**
 * @brief Find object on single frame (only console output)
 *
 * @param ar AR instance
 * @param frame
 * @return
 */
int single_wsl_test(AR *ar, const cv::Mat &frame);

int main(int, char **) {
    // init AR instance
    AR *ar = new ARMarkerless();
    // load marker images
    cv::Mat mat_1 = cv::imread(R"(../resources/marker/miku.jpg)", 0);
    cv::Mat mat_3 = cv::imread(R"(../resources/marker/czech.jpg)", 0);
    cv::Mat mat_4 = cv::imread(R"(../resources/marker/314.png)", 0);
    cv::Mat mat_2 = cv::imread(R"(../resources/4.jpg)");
    ar->add(mat_1);
    ar->add(mat_3);
    ar->add(mat_4);
    // start AR process

//    single(ar, mat_2);
    single(ar, mat_2);
//    start(ar);


    return 0;
}

int single_wsl_test(AR *ar, const cv::Mat &frame) {
    ar->init(frame.size(), INT_MAX);
    auto startTime = std::chrono::steady_clock::now();
    std::vector<QueryItem> result = ar->process(frame);
    std::cout << "Elapsed time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime)).count() << "ms" << std::endl;

    if (!result.empty()) {
        for (const auto &r: result) {
            std::vector<cv::Point2f> objPose;
            std::cout << "Matched: imgId:" << r.imgId << std::endl;
            objPose = r.objPose;
            std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
        }
    }
    return 0;
}

int single(AR *ar, cv::Mat frame) {
    auto startTime = std::chrono::steady_clock::now();

    std::vector<QueryItem> result = ar->process(frame);
    std::cout << "Elapsed time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime)).count() << "ms" << std::endl;
    if (!result.empty()) {
        for (const auto &r: result) {
            std::vector<cv::Point2f> objPose;

            std::cout << "Matched: imgId:" << r.imgId << std::endl;
            objPose = r.objPose;
            std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;

            cv::Scalar val(255);
            cv::line(frame, objPose[3], objPose[0], val, 3);
            for (int i = 0; i < 3; i++) {
                line(frame, objPose[i], objPose[i + 1], val, 3);
            }

            cv::Point center((objPose[0] + objPose[2]) / 2);

            cv::putText(frame, std::to_string(r.imgId), center, OPENCV_FONT, 3, val, 3);
        }
    }
    cv::namedWindow("ARMarkerless", WINDOW_NORMAL);
    cv::resizeWindow("ARMarkerless", 600, 600);
    cv::imshow("ARMarkerless", frame);
    cv::waitKey();
    return 0;
}

int start(AR *ar) {
    cv::VideoCapture videoCapture;
    videoCapture.open(0);
    if (!videoCapture.isOpened()) {
        std::cout << "Failed to Open Camera" << std::endl;
        return -1;
    }
    cv::Mat frame;

    for (;;) {
        videoCapture >> frame;

        std::vector<QueryItem> result = ar->process(frame);
        if (!result.empty()) {
            for (const auto &r: result) {
                std::vector<cv::Point2f> objPose;
//                std::cout << "Matched: imgId:" << r.imgId << std::endl;
                objPose = r.objPose;
//                std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;

                cv::Scalar val(255);
                cv::line(frame, objPose[3], objPose[0], val);
                for (int i = 0; i < 3; i++) {
                    line(frame, objPose[i], objPose[i + 1], val);
                }

                cv::Point center((objPose[0] + objPose[2]) / 2);

                cv::putText(frame, std::to_string(r.imgId), center, OPENCV_FONT, 6, val, 6);
            }
        }

        cv::namedWindow("ARMarkerless", WINDOW_NORMAL);
        cv::resizeWindow("ARMarkerless", 600, 600);
        cv::imshow("ARMarkerless", frame);
        if (waitKey(1) == 27)
            break; //quit on ESC button
    }

    return 0;
}