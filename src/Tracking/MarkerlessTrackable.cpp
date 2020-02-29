#include "MarkerlessTrackable.hpp"

MarkerlessTrackable::~MarkerlessTrackable() = default;

MarkerlessTrackable::MarkerlessTrackable(const int id, const cv::Mat &descriptors,
                                         const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size)
        : Trackable(id, size) {
    this->descriptors = descriptors;
    this->keyPoints = keyPoints;
    this->numFeatures = keyPoints.size();
}
