#include "MarkerlessTrackable.hpp"

MarkerlessTrackable::~MarkerlessTrackable() {

}

MarkerlessTrackable::MarkerlessTrackable(const int id, const cv::Mat &descriptors,
                                         const std::vector<cv::KeyPoint> &keyPoints, const cv::Size &size)
        : TrackableEntity(id) {
    this->descriptors = descriptors;
    this->keyPoints = keyPoints;
    this->size = size;
    this->numFeatures = keyPoints.size();
}
