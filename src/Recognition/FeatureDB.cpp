#include "FeatureDB.hpp"

FeatureDB::FeatureDB() {
//    descriptorMatcher = cv::DescriptorMatcher::create("FlannBased");
//    descriptorMatcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20, 10, 2));
    descriptorMatcher = new cv::BFMatcher(cv::NORM_HAMMING);
    RADIUS = 0.8f;
}

void FeatureDB::addFeatures(const cv::Mat &feature) {
    descriptorMatcher->add(std::vector<cv::Mat>{feature});
}

void FeatureDB::clear() {
    descriptorMatcher->clear();
}

FeatureDB::~FeatureDB() {
    clear();
}

void FeatureDB::create(int amountCluster) {
    if (amountCluster > 0) {
        //TODO: k-means clustering
    }
    descriptorMatcher->train();
}

void FeatureDB::search(const cv::Mat &feature, std::vector<int> &ids) {
    int size = feature.size().height;
    std::vector<std::vector<cv::DMatch>> matchId;
    descriptorMatcher->knnMatch(feature, matchId, KNN_SIZE);

    for (int i = 0; i < size; ++i) {
        cv::DMatch matchM = matchId[i][0];
        cv::DMatch matchN = matchId[i][1];
        if (matchM.distance < matchN.distance * RADIUS) {
            ids.push_back(matchM.trainIdx);
        } else {
            ids.push_back(-1);
        }
    }
}

int FeatureDB::size() const {
    std::vector<cv::Mat> descriptor = descriptorMatcher->getTrainDescriptors();
    int num = 0;
    std::for_each(descriptor.begin(), descriptor.end(),
                  [&num](const cv::Mat &mat) {
                      num += mat.rows;
                  }
    );
    return num;
}

