#include <iostream>
#include "Tracker.hpp"

Tracker::Tracker() {
    maxAmountCorners = 80;
    minQualityCorners = .1;
    minDistanceCorners = 5;
    MIN_FEATURE_POINTS = 6;
}

void Tracker::start(const cv::Mat &frame, const Boundary &pos) {
    frame.copyTo(prevFrame);
    objectPosition = pos;
    cv::Mat mask = CvUtils::createMask(frame.size(), pos);
    // https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    goodFeaturesToTrack(frame, corners, maxAmountCorners, minQualityCorners, minDistanceCorners, mask);
    prevPyr.clear();
    opticalFlowStatus.clear();
}

bool Tracker::keepTracking(const cv::Mat &frame) {
    std::vector<cv::Point2f> nextCorners;
    std::vector<float> err;

    if (corners.empty())
        return false;

    if (prevPyr.empty())
        cv::buildOpticalFlowPyramid(prevFrame, prevPyr, OPTICAL_FLOW_WINDOW_SIZE, OPTICAL_FLOW_PYRO_MAX_LEVEL);

    cv::buildOpticalFlowPyramid(frame, nextPyr, OPTICAL_FLOW_WINDOW_SIZE, OPTICAL_FLOW_PYRO_MAX_LEVEL);

    cv::calcOpticalFlowPyrLK(prevPyr, nextPyr, corners, nextCorners, opticalFlowStatus, err, OPTICAL_FLOW_WINDOW_SIZE,
                             OPTICAL_FLOW_PYRO_MAX_LEVEL);

    std::vector<cv::Point2f> trackedPrePts;
    std::vector<cv::Point2f> trackedPts;

    int found = 0;
    for (size_t i = 0; i < opticalFlowStatus.size(); i++) {
        //     point tracked
        if (opticalFlowStatus[i] && corners.size() > i &&
            norm(nextCorners[i] - corners[i]) <= 100) {
            found++;
            trackedPrePts.push_back(corners[i]);
            trackedPts.push_back(nextCorners[i]);
        }
    }

    // enough tracking points
    if (found > MIN_FEATURE_POINTS) {
        homography = cv::findHomography(cv::Mat(trackedPrePts), cv::Mat(trackedPts), cv::RANSAC, RANSAC_THRESHOLD);
        if (cv::countNonZero(homography) == 0) {
            std::cout << "Zero homo" << std::endl;
            return false;
        }
        // Calc object position on frame
        std::vector<cv::Point2f> nextObjPos;
        cv::perspectiveTransform(objectPosition, nextObjPos, homography);

        if (!ALLOW_TRACK_POINT_OUTSIDE_FRAME)
            // Check if points outside frame
            if (!CvUtils::ptsInsideFrame(prevFrame.size(), nextObjPos)) {
                std::cout << "pts outside frame" << std::endl;
                return false;
            }

        // Check if detected boundary is a rect
        if (!CvUtils::proveRect(nextObjPos)) {
            std::cout << "not a rect" << std::endl;
            return false;
        }
        // Check optical flow consistence
        if (nextCorners.size() != opticalFlowStatus.size()) {
            std::cout << "flow error" << std::endl;
            return false;
        }

        // Count good points (with status == 1) inside object boundary
        int featurePtInsideRect = CvUtils::amountGoodPtInsideRect(nextCorners, nextObjPos, opticalFlowStatus);

        if (featurePtInsideRect < MIN_FEATURE_POINTS) {
            std::cout << "not enough points" << std::endl;
            return false;
        }

        // Prepare to next frame
        frame.copyTo(prevFrame);
        prevPyr.swap(nextPyr);
        corners = trackedPts;
        objectPosition = nextObjPos;

        return true;
    }

    return false;
}

Tracker::~Tracker()
= default;