
#ifndef AR_CORE_MARKERLESSTRACKABLE_HPP
#define AR_CORE_MARKERLESSTRACKABLE_HPP

#include <opencv2/core/mat.hpp>
#include "TrackableEntity.hpp"

class MarkerlessTrackable : TrackableEntity {
private:
    /*********Features**************/
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    /********************************/
    int numFeatures; // number of features
    cv::Size size;   // size of marker image

    friend class MarkerlessDB;

public:
    MarkerlessTrackable(const int id, const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keyPoints,
                        const cv::Size &size);

    ~MarkerlessTrackable();
};


#endif //AR_CORE_MARKERLESSTRACKABLE_HPP
