#pragma once

#ifndef __CVUTILS__
#define __CVUTILS__

#include <opencv2/opencv.hpp>
#include <vector>
#include "base64/base64.hpp"

/*
 * Boundary described by 4 points
 *
 * [0]: Top-Left
 * [1]: Bottom-Left
 * [2]: Bottom-Right
 * [3]: Top-Right
 */
typedef std::vector<cv::Point2f> Boundary;

/**
 * Masking Strategy
 */
enum MaskingStrategy {
    MASK_FAST,      // simply mask roi
    MASK_ACCURACY   // mask with polygon
};

/**
 * QueryItemStatus
 */
enum QueryItemStatus {
    NONE,       // default state
    DETECTED,   // has been detected on frame
    TRACKED,    // detected => tracked
    LOST        // if tracked was lost
};

/**
 * @brief Store query results
 * 
 */
struct QueryItem {
    int imgId;                        // id of marker
    cv::Size imgSize;                 // size of image
    double probability;               // match probability
    int amountMatched;                // amount feature matches
    std::vector<cv::Point2f> objPose; // position of object
    std::vector<cv::Point2f> scaledObjPose; // scaled position of object
    cv::Mat homography;               // homography matrix
    QueryItemStatus status = NONE;           // status


    inline bool operator<(const QueryItem &a) {
        return (probability < a.probability);
    }

    inline bool operator>(const QueryItem &a) {
        return (probability > a.probability);
    }
};

/**
 * @brief CvUtils class
 * Provides some utility functions with Mat object
 */
class CvUtils {
public:
    /**
     * @brief Create a mask with specified boundary
     * 
     * @param size - size of source mat
     * @param boundary - boundary of mask
     * @param strategy - strategy of masking
     * @param color - mask color (default = 255)
     * @return cv::Mat - masked mat
     */
    static cv::Mat createMask(const cv::Size &size, const Boundary &boundary, MaskingStrategy strategy = MASK_FAST,
                              cv::Scalar color = cv::Scalar(255)) {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        if (strategy == MASK_ACCURACY)
            cv::fillConvexPoly(mask, convertVecType<cv::Point>(boundary), color, cv::LINE_AA, 0);
        else if (strategy == MASK_FAST) {
            float x_min = FLT_MAX, y_min = FLT_MAX,
                    x_max = 0, y_max = 0;
            for (const auto &b: boundary) {
                if (b.x > x_max)
                    x_max = b.x;

                if (b.x < x_min)
                    x_min = b.x;

                if (b.y > y_max)
                    y_max = b.y;

                if (b.y < y_min)
                    y_min = b.y;
            }

            x_min = std::max(0.f, std::min(x_min, (float) size.width));
            y_min = std::max(0.f, std::min(y_min, (float) size.height));

            x_max = std::min((float) size.width, std::max(x_max, 0.f));
            y_max = std::min((float) size.height, std::max(y_max, 0.f));

            mask(cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min)) = color;
        }
        return mask;
    }

    /**
     * @brief Vector of points to Mat object
     * 
     * @param v Point`s vector
     * @return cv::Mat Mat object
     */
    static cv::Mat pointsToMat(const Boundary &v) {
        size_t size = v.size();
        cv::Mat ret(3, size, CV_64FC1);

        for (size_t i = 0; i < size; ++i) {
            ret.at<double>(0, i) = (double) v[i].x;
            ret.at<double>(1, i) = (double) v[i].y;
            ret.at<double>(2, i) = 1.;
        }
        return ret;
    }

    /**
     * @brief Convert type of vector: from <F> to <T>
     * 
     * @tparam T Out type
     * @tparam F Input type
     * @param v Vector of objects
     * @return std::vector<T> 
     */
    template<typename T, typename F>
    static std::vector<T> convertVecType(const std::vector<F> &v) {
        std::vector<T> ret;
        std::transform(v.begin(), v.end(),
                       std::back_inserter(ret),
                       [](const F &p) { return (T) p; });
        return ret;
    }

    /**
     * @brief Check points inside of area
     * 
     * @param size Size of area
     * @param pts Vector of points
     * @return true All points inside area
     * @return false One ore more points outside of area
     */
    static bool ptsInsideFrame(const cv::Size &size, const std::vector<cv::Point2f> &pts) {
        return pts.end() == std::find_if(pts.begin(), pts.end(),
                                         [size](const cv::Point2f &p) -> bool {
                                             return p.x < 0 || p.x > size.width || p.y < 0 || p.y > size.height;
                                         });
    }

    /**
     * @brief Check if points are Rect
     * 
     * @param pts Vector of points
     * @return true Its a rectangle
     * @return false Its not a rectangle
     */
    static bool proveRect(std::vector<cv::Point2f> &rect_pts) {
        CV_Assert(rect_pts.size() == 4);

        bool result_f = true;
        float vec[4][2];
        int i;

        vec[0][0] = rect_pts[1].x - rect_pts[0].x;
        vec[0][1] = rect_pts[1].y - rect_pts[0].y;
        vec[1][0] = rect_pts[2].x - rect_pts[1].x;
        vec[1][1] = rect_pts[2].y - rect_pts[1].y;
        vec[2][0] = rect_pts[3].x - rect_pts[2].x;
        vec[2][1] = rect_pts[3].y - rect_pts[2].y;
        vec[3][0] = rect_pts[0].x - rect_pts[3].x;
        vec[3][1] = rect_pts[0].y - rect_pts[3].y;

        int s;
        float val = vec[3][0] * vec[0][1] - vec[3][1] * vec[0][0];
        if (val > 0)
            s = 1;
        else
            s = -1;

        for (i = 0; i < 3; i++) {
            val = vec[i][0] * vec[i + 1][1] - vec[i][1] * vec[i + 1][0];
            if (val * s <= 0) {
                result_f = false;
                break;
            }
        }

        return result_f;
    }

    /**
     * @brief Count number of points inside a rectangle
     * with good tracking status (=1)
     * 
     * @param points Vector of points
     * @param corner_pts Vector of corners
     * @param status Vector of status
     * @return int Number of good points
     */
    static int
    amountGoodPtInsideRect(const std::vector<cv::Point2f> &pts, const std::vector<cv::Point2f> &corners,
                           std::vector<uchar> &status) {
        if (pts.size() != status.size() || corners.size() != 4)
            return -1;

        int count = 0, idx = 0;
        std::for_each(pts.begin(), pts.end(), [&idx, &count, &corners, &status](const cv::Point2f &pt) {
            if (status[idx] == 1) {
                bool isInside = cv::pointPolygonTest(corners, pt, false) != -1;
                if (isInside)
                    count++;
                else
                    status[idx] = 0;
            }
            idx++;
        });

        return count;
    }

    /**
     * @brief Calculates coordinates of detected object
     * 
     * @param size Size of marker
     * @param mat Homography matrix
     * @return std::vector<cv::Point2f> Coordinates of detected object
     */
    static std::vector<cv::Point2f> transformMarkerCoordToObjectCoord(const cv::Size &size, const cv::Mat &mat) {
        std::vector<cv::Point2f> points;
        float width = (float) (size.width) - 1;
        float height = (float) (size.height) - 1;
        // Create a 3x4 Matrix
        cv::Mat srcMat = (cv::Mat_<double>(3, 4) << 0, 0, width, width,
                0, height, height, 0,
                1, 1, 1, 1);
        cv::Mat destMat = mat * srcMat;
        cv::Point2f pt;
        for (int i = 0; i < 4; i++) {
            pt.x = (float) (destMat.at<double>(0, i) / destMat.at<double>(2, i));
            pt.y = (float) (destMat.at<double>(1, i) / destMat.at<double>(2, i));
            points.push_back(pt);
        }

        return points;
    }

    /**
     * @brief Scale points
     * 
     * @param point_vec Input vector of points
     * @param scale Scale factor
     * @return std::vector<cv::Point2f> Vector of scaled points
     */
    static std::vector<cv::Point2f> scalePoints(std::vector<cv::Point2f> &point_vec, double scale) {
        std::vector<cv::Point2f> ret_vec;

        auto itr = point_vec.begin();
        while (itr != point_vec.end()) {
            ret_vec.push_back(*itr * scale);
            itr++;
        }
        return ret_vec;
    }

    /**
     * @brief calculates the CDF
     *BIN(x, n, p) = n!/(x!*(n-x)!) p^x (1-p)^(n-x)
     *
     * @param x Random variable
     * @param n Total number of trials
     * @param p Probability of success of a single trial
     * @return float CDF value
     */
    static float binomialCDF(int x, int n, float p) {
        if (p == 0 || p > 1 || p < 0) return 0.f;
        float cdf = 0.f;
        float b = 0.f;
        float logP = log(p);
        float logNP = log(1.f - p);
        for (int i = 0; i <= x; ++i) {
            if (i > 0) {
                b += log(n - i + 1) - log(i);
            }
            float logPMF = b + (float) i * logP + (float) (n - i) * logNP;
            cdf += exp(logPMF);
        }
        return cdf;
    }

    /**
     * @brief decode mat from base64 string
     * @param encoded base64 string
     * @return decoded image
     */
    static cv::Mat decodeMat(const std::string &encoded) {
        std::string dec_jpg = base64_decode(encoded);
        std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
        cv::Mat img = cv::imdecode(cv::Mat(data), 0);
        return img;
    }

    /**
     * @brief encode mat to base64 string
     * @param img source image
     * @return encoded string
     */
    static std::string encodeMat(const cv::Mat &img) {
        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        std::string encoded = base64_encode(enc_msg, buf.size());
        return encoded;
    }

};

#endif // __CVUTILS__