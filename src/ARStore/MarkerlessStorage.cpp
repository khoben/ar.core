#include "MarkerlessStorage.hpp"

MarkerlessStorage::MarkerlessStorage() {
    featureDetector = cv::AKAZE::create();
    (featureDetector.dynamicCast<cv::AKAZE>())->setThreshold(3e-3);
    descriptorMatcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    markerAmount = 0;
    RADIUS = 0.8f;
}

void MarkerlessStorage::addFeatures(const cv::Mat &feature) {
    descriptorMatcher->add(std::vector<cv::Mat>{feature});
}

MarkerlessStorage::~MarkerlessStorage() = default;


int MarkerlessStorage::size() const {
    int num = 0;
    std::for_each(markers.begin(), markers.end(),
                  [&num](auto const &e) {
                      num += e->numFeatures;
                  }
    );
    return num;
}

void MarkerlessStorage::add(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keyPoints,
                            const cv::Size &size) {
    addFeatures(descriptors);
    int id = markerAmount;
    markers.push_back(std::make_unique<MarkerlessTrackable>(id, descriptors, keyPoints, size));
    markerAmount++;
}

void MarkerlessStorage::add(const cv::Mat &image) {
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;
    extractFeatures(image, keyPoints, descriptor);
    add(descriptor, keyPoints, image.size());
}

void MarkerlessStorage::extractFeatures(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptor) {
    featureDetector->detectAndCompute(img, cv::noArray(), keyPoints, descriptor);
}

std::vector<QueryItem>
MarkerlessStorage::match(const cv::Mat &queryImage, int minNumMatch, float minProbability) {
    std::vector<QueryItem> results;
    QueryItem result;
    // Extract features from query image
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    extractFeatures(queryImage, keyPoints, descriptors);
    // Search matches
    if (markerAmount < 1)
        return results;

    int size = descriptors.size().height; // number of query image features

    std::vector<std::vector<std::pair<int, int>>> descriptorMatches(markerAmount);

    // iterate over all available markers
    // and find matches
    for (const auto &marker: markers) {
        std::vector<std::vector<cv::DMatch>> matchId;
        descriptorMatcher->knnMatch(descriptors, marker->descriptors, matchId, KNN_SIZE);
        for (int i = 0; i < size; ++i) {
            cv::DMatch matchM = matchId[i][0];
            cv::DMatch matchN = matchId[i][1];
            if (matchM.distance < matchN.distance * RADIUS) {
                descriptorMatches[marker->id].push_back(std::make_pair(matchM.trainIdx, matchN.queryIdx));
            }
        }
    }

    int thresholdDist = (int) round(sqrt(5e-3f * queryImage.size().width * queryImage.size().height / M_PI) + 0.5);
    float pp, prob;
    int markerIdx, numMatches;
    cv::Mat homography;
    std::vector<cv::Point2f> r, q;
    const int amountFeatures = this->size();
    auto markerIt = descriptorMatches.begin();
    while (markerIt != descriptorMatches.end()) {
        markerIdx = (markerIt - descriptorMatches.begin());
        numMatches = markerIt->size();
        if (numMatches > minNumMatch) {
            pp = std::min(0.05f * markers[markerIdx]->numFeatures / amountFeatures, 1.f);
            prob = CvUtils::binomialCDF(numMatches, markers[markerIdx]->numFeatures, pp);

            if (prob >= minProbability) {

                auto matchIt = (*markerIt).begin();
                while (matchIt != (*markerIt).end()) {
                    q.push_back(keyPoints[matchIt->second].pt);
                    r.push_back(markers[markerIdx]->keyPoints[matchIt->first].pt);
                    matchIt++;
                }

                homography = cv::findHomography(cv::Mat(r), cv::Mat(q), cv::RANSAC, thresholdDist);

                if (!homography.empty()) {
                    std::vector<cv::Point2f> posePoint = CvUtils::transformMarkerCoordToObjectCoord(
                            markers[markerIdx]->size,
                            homography);
                    if (CvUtils::_proveRect(posePoint)) {
                        result.imgId = markerIdx;
                        result.amountMatched = numMatches;
                        result.imgSize = markers[markerIdx]->size;
                        result.probability = prob;
                        result.homography = homography;
                        result.objPose = posePoint;
                        result.status = DETECTED;
                        results.push_back(result);
                    }
                }
            }
        }
        r.clear();
        q.clear();
        markerIt++;
    }

    // desc sort by probability value
    if (results.size() > 1)
        std::sort(results.begin(), results.end(), std::greater<>());

    return results;
}

std::vector<QueryItem> MarkerlessStorage::match(const cv::Mat &queryImage) {
    return match(queryImage, MIN_NUM_MATCH, MIN_PROBABILITY);
}




