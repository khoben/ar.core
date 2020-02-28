
#ifndef AR_CORE_MARKERLESSTRACKABLE_HPP
#define AR_CORE_MARKERLESSTRACKABLE_HPP

#include "Trackable.hpp"

class MarkerlessTrackable : Trackable {
private:
    /*********Features**************/
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    /********************************/
    int numFeatures; // number of features
    friend class MarkerlessStorage;

public:
    MarkerlessTrackable(int id, const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keyPoints,
                        const cv::Size &size);

    ~MarkerlessTrackable();
};


#endif //AR_CORE_MARKERLESSTRACKABLE_HPP
