#include "Recognition.hpp"

Recognition::Recognition()
{
    //TODO: init detector and descriptor
    vw = new FeatureDB();
    featureDetector = cv::AKAZE::create();
    //    featureDetector = cv::xfeatures2d::SURF::create();
    imageAmount = 0;
    featureAmount = 0;
}

void Recognition::addAndCreateBagOfVisualWords(const std::vector<cv::Mat> &imgs, int numClusters)
{
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    for (const auto &img : imgs)
    {
        extractFeatures(img, keyPoints, descriptor);
        vw->addFeatures(descriptor);
    }
    vw->create(numClusters);
}

void Recognition::addVisualWord(const cv::Mat &img)
{
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    extractFeatures(img, keyPoints, descriptor);
    vw->addFeatures(descriptor);
}

void Recognition::createBagOfVisualWords(int numClusters)
{
    vw->create(numClusters);
}

void Recognition::extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor)
{
    featureDetector->detectAndCompute(img, cv::noArray(), keyPoints, descriptor);
}

int Recognition::addTrackImage(const cv::Mat &img)
{
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    extractFeatures(img, keyPoints, descriptor);
    std::vector<int> ids;
    getFeatureIds(descriptor, ids);
    int id = imageAmount;
    int status = storeImageFeatures(id, img.size(), keyPoints, ids);

    if (status < 0)
        return -1;
    return 0;
}

int Recognition::getFeatureIds(const cv::Mat &descriptor, std::vector<int> &ids)
{
    if (descriptor.empty())
        return -1;
    int size = descriptor.rows;

    cv::Mat id = vw->search(descriptor);

    for (int i = 0; i < size; ++i)
    {
        ids.push_back(id.at<int>(i, 0));
    }

    return 0;
}

int Recognition::storeImageFeatures(int id, const cv::Size &size, std::vector<cv::KeyPoint> keyPoints,
                                    std::vector<int> ids)
{
    FeatureInfo fInfo;

    ImageInfo iInfo;
    iInfo.numFeatures = keyPoints.size();
    iInfo.size = size;

    // Image info
    auto statusAdd = imageInfoStore.insert(std::pair<int, ImageInfo>(id, iInfo));

    if (!statusAdd.second)
        return -1;

    // Vote info
    auto *voteTable = new std::vector<FeatureVote>;
    voteStorage.insert(std::pair<int, std::vector<FeatureVote> *>(id, voteTable));

    fInfo.imgId = id;
    int kpId, kpSize = keyPoints.size();

    for (int i = 0; i < kpSize; ++i)
    {
        if (ids[i] >= 0)
        {
            kpId = getCandidateKpId();
            fInfo.keyPointId = kpId;
            keyPointStore.insert(std::pair<int, cv::KeyPoint>(kpId, keyPoints[i]));
            featureStore.insert(std::pair<int, FeatureInfo>(ids[i], fInfo));
        }
    }

    imageAmount++;
    return 0;
}

std::vector<QueryItem> Recognition::queryImage(const cv::Mat &img, int amountRes)
{
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

int Recognition::getCandidateKpId()
{
    int size = keyPointStore.size();
    if (featureAmount == size)
    {
        featureAmount++;
        return featureAmount;
    }
    else if (featureAmount > size)
    {
        for (int i = 1; i <= featureAmount; ++i)
        {
            if (keyPointStore.count(i) == 0)
            {
                return i;
            }
        }
    }
    return 0;
}

std::vector<QueryItem>
Recognition::searchImageId(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids, cv::Size size, int amountWords,
                           int amountRes)
{
    std::vector<QueryItem> queryReturn;
    voteQueryFeatures(keyPoints, ids);
    std::vector<QueryItem> tmp = getMatchResults(keyPoints, amountWords);
    queryReturn = filterGeomResults(keyPoints, tmp, size, amountRes);
    clearVote();
    return queryReturn;
}

void Recognition::voteQueryFeatures(std::vector<cv::KeyPoint> keyPoints, std::vector<int> ids)
{
    std::multimap<int, FeatureInfo>::iterator featureIt;
    std::map<int, std::vector<FeatureVote> *>::iterator voteIt;
    FeatureVote vote;
    FeatureInfo feature;

    int size = keyPoints.size(), idsIdx = 0;

    for (int i = 0; i < size; ++i)
    {
        if (ids[idsIdx] >= 0)
        {
            featureIt = featureStore.find(ids[idsIdx]);
            while (featureIt != featureStore.end() && featureIt->first == ids[idsIdx])
            {
                feature = featureIt->second;
                vote.featureId = i;
                vote.keyPointId = feature.keyPointId;
                voteIt = voteStorage.find(feature.imgId);
                if (voteIt != voteStorage.end())
                {
                    voteIt->second->push_back(vote);
                }
                featureIt++;
            }
        }
        idsIdx++;
    }
}

std::vector<QueryItem> Recognition::getMatchResults(std::vector<cv::KeyPoint> keyPoints, int amountWords)
{
    std::vector<QueryItem> queryResult;
    QueryItem queryItem;
    std::vector<FeatureVote> *voteTable;
    int numMatch, imgId, imgFeatureNum, featureNum = keyPoints.size();
    float pp, prob;
    auto it = voteStorage.begin();
    while (it != voteStorage.end())
    {
        voteTable = it->second;
        numMatch = voteTable->size();
        if (numMatch >= MIN_MATCH)
        {
            imgId = it->first;
            imgFeatureNum = imageInfoStore[imgId].numFeatures;
            pp = std::min((float)imgFeatureNum / amountWords, 1.f);
            prob = probDistribution(featureNum, numMatch, pp);

            if (prob >= MIN_PROBABILITY)
            {
                queryItem.imgId = imgId;
                queryItem.amountMatched = numMatch;
                queryItem.imgSize = imageInfoStore[imgId].size;
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
                               int amountRes)
{
    std::vector<QueryItem> queryResult;
    QueryItem queryItem;
    int sizePre = pre.size();
    if (sizePre < 1)
        return queryResult;

    int imgId, count = 0;
    std::vector<FeatureVote> *voteTable;
    std::vector<cv::Point2f> q, r;
    cv::Mat homography;
    int thresholdDist = (int)round(sqrt(DISTANTION_TOLERANCE * size.width * size.height / M_PI));
    for (int i = 0; i < sizePre && count < amountRes; ++i)
    {
        queryItem = pre[i];
        imgId = queryItem.imgId;

        voteTable = voteStorage[imgId];
        findPointPair(keyPoints, *voteTable, q, r);
        homography = cv::findHomography(cv::Mat(r), cv::Mat(q), cv::RANSAC, thresholdDist);
        std::vector<cv::Point2f> posePoint = CvUtils::transformMarkerCoordToObjectCoord(imageInfoStore[imgId].size,
                                                                                        homography);

        if (CvUtils::_proveRect(posePoint))
        {
            queryItem.homography = homography;
            queryItem.objPose = posePoint;
            queryResult.push_back(queryItem);
            count++;
        }

        r.clear();
        q.clear();
    }

    return queryResult;
}

void Recognition::clearVote()
{
    auto it = voteStorage.begin();
    while (it != voteStorage.end())
    {
        it->second->clear();
        it++;
    }
}

float Recognition::probDistribution(int numFeatures, int numMatch, float pp)
{
    if (pp == 1)
        return pp;
    float prob = 0.f;
    float logPp = log(pp);
    float logNp = log(1 - pp);
    float tmp;
    for (int i = 0; i <= numMatch; ++i)
    {
        tmp = 0.f;
        for (int j = 0; j < i; ++j)
        {
            tmp += log(numFeatures - j) - log(j + 1);
        }
        tmp += logPp * i + logNp * (numFeatures - i);
        prob += exp(tmp);
        if (prob > 1)
        {
            prob = 1;
            break;
        }
    }
    return prob;
}

void Recognition::findPointPair(std::vector<cv::KeyPoint> keyPoints, std::vector<FeatureVote> voteTable,
                                std::vector<cv::Point2f> &q, std::vector<cv::Point2f> &r)
{
    auto it = voteTable.begin();
    while (it != voteTable.end())
    {
        q.push_back(keyPoints[it->featureId].pt);
        r.push_back(keyPointStore[it->keyPointId].pt);
        it++;
    }
}

Recognition::~Recognition() = default;