#include "BoVw.hpp"

BoVW::BoVW() {
//    descriptorMatcher = cv::DescriptorMatcher::create("FlannBased");
    descriptorMatcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20,10,2));
//    descriptorMatcher = new cv::BFMatcher(cv::NORM_HAMMING);
    VOTE_NUM = 1;
//    RADIUS = 0.2;
    RADIUS = 0.99f;
}

void BoVW::addFeatures(const cv::Mat &feature) {
    descriptorMatcher->add(std::vector<cv::Mat>{feature});
}

void BoVW::clear() {
    descriptorMatcher->clear();
}

BoVW::~BoVW() {
    clear();
}

void BoVW::create(int amountCluster) {
    if (amountCluster > 0) {
        //TODO: k-means clustering
    }
    descriptorMatcher->train();
}

cv::Mat BoVW::search(const cv::Mat &feature) {
    const int KNN_SIZE = VOTE_NUM + 1;
    int size = feature.size().height;

    cv::Mat id(size, KNN_SIZE, CV_32SC1);
    std::vector<std::vector<cv::DMatch>> matchId;
    descriptorMatcher->knnMatch(feature, matchId, KNN_SIZE);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < VOTE_NUM; ++j) {
            cv::DMatch matchM = matchId[i][0];
            cv::DMatch matchN = matchId[i][1];
            if (matchM.distance >= RADIUS * matchN.distance) {
                id.at<int>(i, j) = -1;
            } else {
                id.at<int>(i, j) = matchM.trainIdx;
            }
        }
    }

    return id;
}

int BoVW::size() const {
    std::vector<cv::Mat> descriptor = descriptorMatcher->getTrainDescriptors();
    int num = 0;
    std::for_each(descriptor.begin(), descriptor.end(),
                  [&num](const cv::Mat &mat) {
                      num += mat.rows;
                  }
    );
    return num;
}

