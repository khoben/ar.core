#include "Tracking.hpp"

Tracking::Tracking()
{
    maxAmountCorners = 75;
    minQualityCorners = .1;
    minDistanceCorners = 5;
}

Tracking *Tracking::create()
{
    return new Tracking();
}

void Tracking::start(const cv::Mat &img, ObjectPosition pos)
{
    img.copyTo(prevFrame);
    objectPosition = pos;
    cv::Mat mask = CvUtils::createMask(img.size(), pos);
    // https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    goodFeaturesToTrack(img, corners, maxAmountCorners, minQualityCorners, minDistanceCorners, mask);
    opticalFlowStatus.clear();
}

bool Tracking::keepTracking(const cv::Mat &img)
{
    std::vector<cv::Point> nextCorners;
    std::vector<float> err;
    // https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    calcOpticalFlowPyrLK(prevFrame, img, corners, nextCorners, opticalFlowStatus, err);

    // check track status
    int found = std::count(opticalFlowStatus.begin(), opticalFlowStatus.end(), 1);

    // enough points
    if (found < 6)
    {
        return false;
    }
    else
    {
        homography = cv::findHomography(cv::Mat(corners), cv::Mat(nextCorners), opticalFlowStatus, CV_RANSAC, 5);
        if (cv::countNonZero(homography) == 0)
        {
            return false;
        }
        else
        {
            // calc object position
        }
    }
}

Tracking::~Tracking()
{
}