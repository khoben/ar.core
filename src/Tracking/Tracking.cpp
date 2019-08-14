#include <iostream>
#include "Tracking.hpp"

Tracking::Tracking() {
    maxAmountCorners = 80;
    minQualityCorners = .1;
    minDistanceCorners = 5;
    MIN_FEATURE_POINTS = 6;
}

Tracking *Tracking::create() {
    return new Tracking();
}

void Tracking::start(const cv::Mat &frame, const ObjectPosition &pos) {
    frame.copyTo(prevFrame);
    objectPosition = pos;
    cv::Mat mask = CvUtils::createMask(frame.size(), pos);
    // https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    goodFeaturesToTrack(frame, corners, maxAmountCorners, minQualityCorners, minDistanceCorners, mask);
    opticalFlowStatus.clear();
}

bool Tracking::keepTracking(const cv::Mat &frame) {
    std::vector<cv::Point2f> nextCorners;
    std::vector<float> err;
    // https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    calcOpticalFlowPyrLK(prevFrame, frame, corners, nextCorners, opticalFlowStatus, err);

    // check track status
    int found = std::count(opticalFlowStatus.begin(), opticalFlowStatus.end(), 1);
    std::cout << "found: " << found << std::endl;
    // enough tracking points
    if (found < MIN_FEATURE_POINTS) {
        return false;
    } else {
        homography = cv::findHomography(cv::Mat(corners), cv::Mat(nextCorners), opticalFlowStatus, cv::RANSAC, 2.5);
        if (cv::countNonZero(homography) == 0) {
            std::cout << "Zero homo" << std::endl;
            return false;
        } else {
            // calc object position
//            std::vector<cv::Point2f> nextObjPos = CvUtils::calcObjPos(objectPosition, homography);
            std::vector<cv::Point2f> nextObjPos;
            cv::perspectiveTransform(objectPosition, nextObjPos, homography);
            cv::Size size = prevFrame.size();
            // True if points inside frame
            if (CvUtils::_ptsInsideFrame(size, nextObjPos)) {
                std::cout << "pts inside" << std::endl;
                return false;
            }
            // True if that`s rect
            if (!CvUtils::_proveRect(nextObjPos)) {
                std::cout << "not a rect" << std::endl;
                return false;
            }
            if (nextCorners.size() != opticalFlowStatus.size()) {
                std::cout << "flow error" << std::endl;
                return false;
            }

            int featurePtInsideRect = CvUtils::_amountGoodPtInsideRect(nextCorners, nextObjPos, opticalFlowStatus);

            if (featurePtInsideRect < MIN_FEATURE_POINTS) {
                std::cout << "not enough points" << std::endl;
                return false;
            }

            frame.copyTo(prevFrame);
            corners = nextCorners;
            objectPosition = nextObjPos;
        }
    }
    return true;
}

Tracking::~Tracking()
= default;