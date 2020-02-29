#ifndef AR_CORE_TRACKABLE_HPP
#define AR_CORE_TRACKABLE_HPP

#include <opencv2/core/mat.hpp>

/**
 * Base class for trackable entity
 */
class Trackable {
protected:
    // id of trackable
    int id;
    // size of marker image
    cv::Size size;

    explicit Trackable(int id, const cv::Size &size);

public:
};

#endif //AR_CORE_TRACKABLE_HPP
