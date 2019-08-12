#include "Recognition.hpp"

Recognition::Recognition() {
    //TODO: init detector and descriptor
    vw = new BoVW();
    featureDetector = cv::AKAZE::create();
}

void Recognition::createBagOfVisualWords(const std::vector<cv::Mat> &imgs, int numClusters) {
    for (const auto &img: imgs) {
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptor;
        extractFeatures(img, keyPoints, descriptor);
        vw->addFeatures(descriptor);
    }
    vw->create(numClusters);
}

void Recognition::extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> keyPoints, cv::Mat &descriptor) {
    featureDetector->detectAndCompute(img, cv::noArray(), keyPoints, descriptor);
}

int Recognition::addTrackImage(const cv::Mat &img, int id) {
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    extractFeatures(img, keyPoints, descriptor);
    std::vector<int> ids;
    getFeatureIds(descriptor, ids);
    int status = storeImageFeatures(id, img.size(), keyPoints, ids);

    if (status < 0) return -1;
    return 0;
}

int Recognition::getFeatureIds(const cv::Mat &descriptor, std::vector<int> &ids) {
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

int Recognition::storeImageFeatures(int id, const cv::Size &size, std::vector<cv::KeyPoint> keyPoints,
                                    std::vector<int> ids) {
    featureInfo fInfo;

    imageInfo iInfo;
    iInfo.feature_num = keyPoints.size();
    iInfo.img_size = size;

    // Image info
    auto statusAdd = imageInfoStore.insert(std::pair<int, imageInfo>(id, iInfo));

    if (!statusAdd.second) return -1;

    // Vote info
    auto *voteTable = new std::vector<featureVote>;
    voteStorage.insert(std::pair<int, std::vector<featureVote> *>(id, voteTable));

    fInfo.img_id = id;
    int kpId, kpSize = keyPoints.size();

    for (int i = 0; i < kpSize; ++i) {
        if (ids[i * vw->getVote()] >= 0) {
            kpId = getCandidateKpId();
            fInfo.keypoint_id = kpId;
            keyPointStore.insert(std::pair<int, cv::KeyPoint>(kpId, keyPoints[i]));
            featureStore.insert(std::pair<int, featureInfo>(ids[i * vw->getVote()], fInfo));
        }
    }

    imageAmount++;
    return 0;
}

std::vector<QueryItem> Recognition::queryImage(const cv::Mat &img, int amountRes) {
    std::vector<QueryItem> queryReturn;
    std::vector<cv::KeyPoint> kp;
    cv::Mat descriptor;

    extractFeatures(img, kp, descriptor);
    std::vector<int> ids;
    int status = getFeatureIds(descriptor, ids);
    if (status < 0)
        return queryReturn;

    queryReturn = vw->searchImageId(kp, ids, img.size(), amountRes);
    return queryReturn;
}

int Recognition::getCandidateKpId() {
    int size = keyPointStore.size();
    if (featureAmount == size) {
        featureAmount++;
        return featureAmount;
    } else if (featureAmount > size) {
        for (int i = 1; i <= featureAmount; ++i) {
            if(keyPointStore.count(i) == 0){
                return i;
            }
        }
    }
    return 0;
}

Recognition::~Recognition() = default;