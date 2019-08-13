#include "Recognition.hpp"

Recognition::Recognition() {
    //TODO: init detector and descriptor
    vw = new BoVW();
    featureDetector = cv::AKAZE::create();
//    featureDetector = cv::xfeatures2d::SIFT::create();
    imageAmount = 0;
    featureAmount = 0;
}

void Recognition::createBagOfVisualWords(const std::vector<cv::Mat> &imgs, int numClusters) {
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    for (const auto &img: imgs) {
        extractFeatures(img, keyPoints, descriptor);
        vw->addFeatures(descriptor);
    }
    vw->create(numClusters);
}

void Recognition::extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint>& keyPoints, cv::Mat &descriptor) {
    featureDetector->detectAndCompute(img, cv::noArray(), keyPoints, descriptor);
}

int Recognition::addTrackImage(const cv::Mat &img) {
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    extractFeatures(img, keyPoints, descriptor);
    std::vector<int> ids;
    getFeatureIds(descriptor, ids);
    int id = imageAmount;
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

    queryReturn = searchImageId(kp, ids, img.size(), vw->size(), amountRes);
    return queryReturn;
}

int Recognition::getCandidateKpId() {
    int size = keyPointStore.size();
    if (featureAmount == size) {
        featureAmount++;
        return featureAmount;
    } else if (featureAmount > size) {
        for (int i = 1; i <= featureAmount; ++i) {
            if (keyPointStore.count(i) == 0) {
                return i;
            }
        }
    }
    return 0;
}

std::vector<QueryItem>
Recognition::searchImageId(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids, cv::Size size, int amountWords,
                           int amountRes) {
    std::vector<QueryItem> queryReturn;
    voteQueryFeatures(keyPoints, ids);
    std::vector<QueryItem> tmp = getMatchResults(keyPoints, amountWords);
    queryReturn = filterGeomResults(keyPoints, tmp, size, amountRes);
    clearVote();
    return queryReturn;
}

void Recognition::voteQueryFeatures(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids) {
    std::multimap<int, featureInfo>::iterator featureIt;
    std::map<int, std::vector<featureVote> *>::iterator voteIt;
    featureVote vote;
    featureInfo feature;

    int size = keyPoints.size(), voteNum = vw->getVote(),
            idsIdx = 0;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < voteNum; ++j) {
            if (ids[idsIdx] >= 0) {
                featureIt = featureStore.find(ids[idsIdx]);
                while (featureIt != featureStore.end() && featureIt->first == ids[idsIdx]) {
                    feature = featureIt->second;
                    vote.in_feat_i = i;
                    vote.keypoint_id = feature.keypoint_id;
                    voteIt = voteStorage.find(feature.img_id);
                    if (voteIt != voteStorage.end()) {
                        voteIt->second->push_back(vote);
                    }
                    featureIt++;
                }
            }
            idsIdx++;
        }
    }
}

std::vector<QueryItem> Recognition::getMatchResults(std::vector<cv::KeyPoint> keyPoints, int amountWords) {

    std::vector<QueryItem> queryResult;
    QueryItem queryItem;
    std::vector<featureVote> *voteTable;
    int numMatch, imgId, imgFeatureNum, vote = vw->getVote(), featureNum = keyPoints.size();
    float pp, prob;
    auto it = voteStorage.begin();
    while (it != voteStorage.end()) {
        voteTable = it->second;
        numMatch = voteTable->size();
        if (numMatch >= MIN_MATCH) {
            imgId = it->first;
            imgFeatureNum = imageInfoStore[imgId].feature_num;
            pp = std::min((float)vote * imgFeatureNum / amountWords, 1.f);
            prob = probDistribution(featureNum, numMatch, pp);

            if (prob >= MIN_PROBABILITY) {
                queryItem.imgId = imgId;
                queryItem.amountMatched = numMatch;
                queryItem.imgSize = imageInfoStore[imgId].img_size;
                queryItem.probability = prob;
                queryResult.push_back(queryItem);
            }
        }
        it++;
    }
    std::sort(queryResult.begin(), queryResult.end(), [](const QueryItem &a, const QueryItem &b) -> bool {
        return a.probability > b.probability;
    });
    return queryResult;
}

std::vector<QueryItem>
Recognition::filterGeomResults(std::vector<cv::KeyPoint> keyPoints, std::vector<QueryItem> pre, cv::Size size,
                               int amountRes) {

    std::vector<QueryItem> queryResult;
    QueryItem queryItem;
    int sizePre = pre.size();
    if (sizePre < 1)
        return queryResult;

    int imgId, numMatch, numFeatures = keyPoints.size(), count = 0;
    std::vector<featureVote> *voteTable;
    std::vector<cv::Point2f> q, r;
    cv::Mat pose;
    int thresholdDist = round(sqrt(DISTANTION_TOLERANCE * size.width * size.height / M_PI));
    for (int i = 0; i < sizePre && count < amountRes; ++i) {
        queryItem = pre[i];
        imgId = queryItem.imgId;
        numMatch = queryItem.amountMatched;

        voteTable = voteStorage[imgId];
        findPointPair(keyPoints, *voteTable, q, r);
        pose = cv::findHomography(cv::Mat(r), cv::Mat(q), cv::RANSAC, thresholdDist);
        std::vector<cv::Point2f> posePoint = CvUtils::affineTransformRect(imageInfoStore[imgId].img_size, pose);

        if (CvUtils::_proveRect(posePoint)) {
            queryItem.pose = pose;
            queryItem.objPose = posePoint;
            queryResult.push_back(queryItem);
            count++;
        }

        r.clear();
        q.clear();
    }

    return queryResult;
}

void Recognition::clearVote() {
    auto it = voteStorage.begin();
    while (it != voteStorage.end()) {
        it->second->clear();
        it++;
    }
}

float Recognition::probDistribution(int numFeatures, int numMatch, float pp) {
    float prob = 0.f;
    float logPp = log(1.f - pp);
    float logNp = log(pp);
    float tmp;
    for (int i = 0; i <= numMatch; ++i) {
        tmp = 0.f;
        for (int j = 0; j < i; ++j) {
            tmp += log(numFeatures - j) - log(j + 1);
        }
        tmp += logPp * i + logNp * (numFeatures - i);
        prob += exp(tmp);
        if (prob > 1) {
            prob = 1;
            break;
        }
    }
    return prob;
}

void Recognition::findPointPair(std::vector<cv::KeyPoint> keyPoints, std::vector<featureVote> voteTable,
                                std::vector<cv::Point2f>& q, std::vector<cv::Point2f>& r) {
    auto it = voteTable.begin();
    while (it != voteTable.end()) {
        q.push_back(keyPoints[it->in_feat_i].pt);
        r.push_back(keyPointStore[it->keypoint_id].pt);
        it++;
    }
}

Recognition::~Recognition() = default;