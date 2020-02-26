#include "opencv2/opencv.hpp"
#include "AR/Ar.hpp"
#include <climits>
#include <chrono>

using namespace cv;

AR *ar;                       // AR instance
const int maxFrameSize = INT_MAX; // maximum query frame size
const int OPENCV_FONT = FONT_HERSHEY_DUPLEX; // OpenCV font

/**
 * @brief Make query resized frame
 * 
 * @param size Size of source frame
 * @param max_size Max size
 * @param scale Initial scale
 * @return cv::Mat Query-resized frame
 */
cv::Mat makeQueryMat(const cv::Size &size, int max_size, int &scale);

/**
 * @brief Find the object on videostream
 * 
 * @return int 
 */
int start();

/**
 * @brief Find object on single frame
 * 
 * @param frame 
 * @return int 
 */
int single(cv::Mat frame);

int single_wsl_test(cv::Mat frame);

int main(int, char **) {
    // init AR instance
    ar = new AR();
    // load marker images
    cv::Mat mat_1 = cv::imread(R"(../resources/marker/miku.jpg)", 0);
    cv::Mat mat_3 = cv::imread(R"(../resources/marker/czech.jpg)", 0);
    cv::Mat mat_4 = cv::imread(R"(../resources/marker/314.png)", 0);
    cv::Mat mat_2 = cv::imread(R"(../resources/6.jpg)");
    ar->add(mat_1);
    ar->add(mat_3);
    ar->add(mat_4);
    // start AR process

    single_wsl_test(mat_2);
//    start();
    return 0;
}

cv::Mat makeQueryMat(const cv::Size &size, int max_size, int &scale) {
    int frame_max_size = std::max(size.width, size.height);
    scale = 1;
    while ((frame_max_size / scale) > max_size) {
        scale *= 2;
    }
    return cv::Mat(size.height / scale, size.width / scale, CV_8UC1);
}

int single_wsl_test(cv::Mat frame) {
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat gray, query;
    int scale = 1;
    query = makeQueryMat(frame.size(), maxFrameSize, scale);

    if (frame.empty())
        return -1;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, query, query.size());
    std::vector<QueryItem> result = ar->process(query);
    std::cout << "Elapsed time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime)).count() << "ms" << std::endl;
    if (!result.empty()) {
        for (const auto &r: result) {
            std::vector<cv::Point2f> objPose;

            std::cout << "Matched: imgId:" << r.imgId << std::endl;
            objPose = r.objPose;
            objPose = CvUtils::scalePoints(objPose, scale);
            std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
        }
    }
    return 0;
}

int single(cv::Mat frame) {
    auto startTime = std::chrono::steady_clock::now();
    cv::Mat gray, query;
    int scale = 1;
    query = makeQueryMat(frame.size(), maxFrameSize, scale);

    int imgId = -1;

    if (frame.empty())
        return -1;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, query, query.size());
    std::vector<QueryItem> result = ar->process(query);
    std::cout << "Elapsed time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime)).count() << "ms" << std::endl;
    if (!result.empty()) {
        for (const auto &r: result) {
            std::vector<cv::Point2f> objPose;

            std::cout << "Matched: imgId:" << r.imgId << std::endl;
            objPose = r.objPose;
            objPose = CvUtils::scalePoints(objPose, scale);
            std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
            imgId = r.imgId;

            cv::Scalar val(255);
            cv::line(frame, objPose[3], objPose[0], val, 3);
            for (int i = 0; i < 3; i++) {
                line(frame, objPose[i], objPose[i + 1], val, 3);
            }

            cv::Point center((objPose[0] + objPose[2]) / 2);

            cv::putText(frame, std::to_string(imgId), center, OPENCV_FONT, 3, val, 3);
        }
    }
    cv::namedWindow("AR", WINDOW_NORMAL);
    cv::resizeWindow("AR", 600, 600);
    cv::imshow("AR", frame);
    cv::waitKey();
    return 0;
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
        if (frame.empty())
            break;
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

                std::cout << "Matched: imgId:" << r.imgId << std::endl;
                objPose = r.objPose;
                objPose = CvUtils::scalePoints(objPose, scale);
                std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
                ar->startTracking(gray, objPose);
                isTracked = true;
                imgId = r.imgId;
            }
        }
        if (isTracked && imgId != -1) {
            cv::Scalar val(255);
            ObjectPosition objPose = ar->getTrackingInstance()->objectPosition;
            cv::line(frame, objPose[3], objPose[0], val);
            for (int i = 0; i < 3; i++) {
                line(frame, objPose[i], objPose[i + 1], val);
            }

            cv::Point center((objPose[0] + objPose[2]) / 2);

            cv::putText(frame, std::to_string(imgId), center, OPENCV_FONT, 6, val, 6);
        }
        cv::namedWindow("AR", WINDOW_NORMAL);
        cv::resizeWindow("AR", 600, 600);
        cv::imshow("AR", frame);
        if (waitKey(1) == 27)
            break; //quit on ESC button
    }

    return 0;
}