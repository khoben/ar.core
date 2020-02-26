#include <iostream>
#include "Tracking.hpp"

Tracking::Tracking() {
    maxAmountCorners = 80;
    minQualityCorners = .1;
    minDistanceCorners = 5;
    MIN_FEATURE_POINTS = 6;
}

void Tracking::start(const cv::Mat &frame, const ObjectPosition &pos) {
    frame.copyTo(prevFrame);
    objectPosition = pos;
    cv::Mat mask = CvUtils::createMask(frame.size(), pos);
    // https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    goodFeaturesToTrack(frame, corners, maxAmountCorners, minQualityCorners, minDistanceCorners, mask);
    prevPyr.clear();
    opticalFlowStatus.clear();
}

bool Tracking::keepTracking(const cv::Mat &frame) {
    std::vector<cv::Point2f> nextCorners;
    std::vector<float> err;

    if (corners.empty())
        return false;

    if (prevPyr.empty())
        cv::buildOpticalFlowPyramid(prevFrame, prevPyr, cv::Size(21, 21), 3, true);

    cv::buildOpticalFlowPyramid(frame, nextPyr, cv::Size(21, 21), 3, true);
    // https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
//    calcOpticalFlowPyrLK(prevFrame, frame, corners, nextCorners, opticalFlowStatus, err);

    cv::calcOpticalFlowPyrLK(prevPyr, nextPyr, corners, nextCorners, opticalFlowStatus, err, cv::Size(21, 21), 3);

    std::vector<cv::Point2f> trackedPrePts;
    std::vector<cv::Point2f> trackedPts;

    int found = 0;
    for (size_t i = 0; i < opticalFlowStatus.size(); i++) {
        if (opticalFlowStatus[i] && corners.size() > i && norm(nextCorners[i] - corners[i]) <=
                                                          100/*&& m_prevPts_twice.size()>i && norm(m_prevPts[i] - m_prevPts_twice[i]) <= 2*/) {
            found++;
            trackedPrePts.push_back(corners[i]);
            trackedPts.push_back(nextCorners[i]);
        }
    }

    // check track status
//    int found = std::count(opticalFlowStatus.begin(), opticalFlowStatus.end(), 1);
//    std::cout << "found: " << found << std::endl;
    // enough tracking points
    if (found < MIN_FEATURE_POINTS) {
        return false;
    } else {
        homography = cv::findHomography(cv::Mat(trackedPrePts), cv::Mat(trackedPts), cv::RANSAC, 10);
        if (cv::countNonZero(homography) == 0) {
            std::cout << "Zero homo" << std::endl;
            return false;
        } else {
            // Calc object position on frame
            std::vector<cv::Point2f> nextObjPos;
            nextObjPos = CvUtils::calcObjPos(objectPosition, homography);
//            cv::perspectiveTransform(objectPosition, nextObjPos, homography);
//            cv::Size size = prevFrame.size();

            // Check if points outside frame
//            if (!CvUtils::ptsInsideFrame(size, nextObjPos)) {
//                std::cout << "pts outside frame" << std::endl;
//                return false;
//            }
            // Check if detected boundary is a rect
            if (!CvUtils::_proveRect(nextObjPos)) {
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
        }
    }
    return true;
}

Tracking::~Tracking()
= default;