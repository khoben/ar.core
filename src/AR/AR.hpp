#ifndef AR_CORE_AR_HPP
#define AR_CORE_AR_HPP

#include "../Tracking/Tracker.hpp"
#include "../Detection/DetectMarkerless.hpp"

/**
 * Base class AR
 */
class AR {
protected:
    // scale factor for queryMat
    float scale;
    // scaled mat for querying
    cv::Mat queryMat;
    // masked mat for querying
    cv::Mat maskedMat;
    // size of queryMat (w x h)
    cv::Size querySize;

    // tracking instances
    std::vector<std::pair<QueryItem, std::unique_ptr<Tracker>>> trackingItems;
    // recognition instance
    std::unique_ptr<Detect> recognitionInstance;

public:
    /**
     * @brief Initialize AR instance: set querying image size
     *
     * @param frameSize source frame size
     * @param maxSize maximum size of querying image in pixels
     * for one of sides
     */
    virtual void init(const cv::Size &frameSize, int maxSize) = 0;

    /**
     * @brief Start recognition&tracking process
     *
     * @param frame Frame
     * @return std::vector<QueryItem> Result
     */
    virtual std::vector<QueryItem> process(const cv::Mat &frame) = 0;

    /**
     * @brief Add marker image
     *
     * @param img Marker image
     * @return int
     */
    virtual int add(const cv::Mat &img) = 0;

    /**
    * @brief Add marker image objects
    *
    * @param imgs marker images
    * @return int
    */
    virtual int addAll(const std::vector<cv::Mat> &imgs) = 0;
};


#endif //AR_CORE_AR_HPP
