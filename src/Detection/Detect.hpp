#ifndef AR_CORE_DETECT_HPP
#define AR_CORE_DETECT_HPP

#include "../ARStore/Storage.hpp"

/**
 * Base class for detection
 */
class Detect {
protected:
    std::unique_ptr<Storage> storage;   // marker storage
public:
    /**
     * @brief Add image to marker database
     *
     * @param img Marker image
     * @return int
     */
    virtual int addTrackImage(const cv::Mat &img) = 0;

    /**
     * @brief Make a query
     *
     * @param img Frame
     * @return std::vector<QueryItem> Query results
     */
    virtual std::vector<QueryItem> queryImage(const cv::Mat &img) = 0;
};


#endif //AR_CORE_DETECT_HPP
