#include "Recognition.hpp"

Recognition::Recognition() {
    //TODO: init detector and descriptor
    vw = new BoVW();
    featureDetector = cv::AKAZE::create();
}

void Recognition::createBagOfVisualWords(const std::vector<cv::Mat>& imgs, int numClusters) {
    for(const auto& img: imgs){
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptor;
        extractFeatures(img, keyPoints, descriptor);
        vw->addFeatures(descriptor);
    }
    vw->create(numClusters);
}

void Recognition::extractFeatures(const cv::Mat& img, std::vector<cv::KeyPoint> keyPoints, cv::Mat& descriptor) {
    featureDetector->detectAndCompute(img, cv::noArray(), keyPoints, descriptor);
}

int Recognition::addTrackImage(const cv::Mat& img, int id) {
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    extractFeatures(img, keyPoints, descriptor);
    std::vector<int> ids;
    getFeatureIds(descriptor, ids);
    int status = storeImageFeatures(id, img.size(), keyPoints, ids);

}

int Recognition::getFeatureIds(const cv::Mat& descriptor, std::vector<int>& ids) {
    if (descriptor.empty()) return -1;
    int vote = vw->getVote();
    int size = descriptor.rows;

    cv::Mat id = vw->search(descriptor);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < vote; ++j) {
            ids.push_back(id.at<int>(i, j));
        }
    }

    return 0;
}

int Recognition::storeImageFeatures(int id, const cv::Size& size, std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids) {

    return 0;
}

int Recognition::queryImage(const cv::Mat &img, int) {
    return 0;
}

Recognition::~Recognition() = default;