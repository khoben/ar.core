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
     * @brief Check points inside of area
     * 
     * @param img_size Size of area
     * @param pts Vector of points
     * @return true All points inside area
     * @return false One ore more points outside of area
     */
    static bool _ptsInsideFrame(const cv::Size &img_size, std::vector<cv::Point2f> &pts) {
        auto itr = pts.begin();
        while (itr != pts.end()) {
            if (itr->x < 0 || itr->x >= img_size.width || itr->y < 0 || itr->y >= img_size.height) {
                return false;
            } else {
                itr++;
            }
        }
        return true;
    }

    /***************HELP FUNCTION FOR PROVERECT(PTS)*****************/
    static bool isOrthogonal(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c) {
        const float eps = 1e-8;
        return fabs((b.x - a.x) * (b.x - c.x) +
                    (b.y - a.y) * (b.y - c.y)) < eps;
    }

    static bool isRect(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c, const cv::Point2f &d) {
        return isOrthogonal(a, b, c) &&
               isOrthogonal(b, c, d) &&
               isOrthogonal(c, d, a);
    }

    static bool isRectAnyOrder(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c, const cv::Point2f &d) {
        return isRect(a, b, c, d) ||
               isRect(b, c, a, d) ||
               isRect(c, a, b, d);
    }
    /*******************************************************************/

    /**
     * @brief Check if points are Rect
     * 
     * @param pts Vector of points
     * @return true Its a rectangle
     * @return false Its not a rectangle
     */
    static bool proveRect(std::vector<cv::Point2f> &pts) {
        if (pts.size() != 4)
            return false;

        return isRectAnyOrder(pts[0], pts[1], pts[2], pts[3]);

    }

    /**
     * @brief Check if points are Rect
     * 
     * @param pts Vector of points
     * @return true Its a rectangle
     * @return false Its not a rectangle
     */
    static bool _proveRect(std::vector<cv::Point2f> &rect_pts) {
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
     * @brief Given three colinear points p, q, r, the function checks if 
     * point q lies on line segment 'pr'
     * 
     * @param p 
     * @param q 
     * @param r 
     * @return true 
     * @return false 
     */
    static bool onSegment(const cv::Point2f &p, const cv::Point2f &q, const cv::Point2f &r) {
        return q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
               q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y);
    }

    /**
     * @brief To find orientation of ordered triplet (p, q, r).
     * The function returns following values
     * 0 --> p, q and r are colinear
     * 1 --> Clockwise
     * 2 --> Counterclockwise
     * 
     * @param p 
     * @param q 
     * @param r 
     * @return int 
     */
    static int orientation(const cv::Point2f &p, const cv::Point2f &q, const cv::Point2f &r) {
        int val = (q.y - p.y) * (r.x - q.x) -
                  (q.x - p.x) * (r.y - q.y);

        if (val == 0)
            return 0;             // colinear
        return (val > 0) ? 1 : 2; // clock or counterclock wise
    }

    /**
     * @brief The function that returns true if line segment 'p1q1'and 'p2q2' intersect.
     * 
     * @param p1 
     * @param q1 
     * @param p2 
     * @param q2 
     * @return true 
     * @return false 
     */
    static bool
    isIntersect(const cv::Point2f &p1, const cv::Point2f &q1, const cv::Point2f &p2, const cv::Point2f &q2) {
        // Find the four orientations needed for general and
        // special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);

        // General case
        if (o1 != o2 && o3 != o4)
            return true;

        // Special Cases
        // p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0 && onSegment(p1, p2, q1))
            return true;

        // p1, q1 and p2 are colinear and q2 lies on segment p1q1
        if (o2 == 0 && onSegment(p1, q2, q1))
            return true;

        // p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0 && onSegment(p2, p1, q2))
            return true;

        // p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0 && onSegment(p2, q1, q2))
            return true;

        return false; // Doesn't fall in any of the above cases
    }

    /**
     * @brief Rectangle with provided points:pt is inside rectangle:rect
     * 
     * @param rect Rectangle
     * @param pt Vector of points test rectangle
     * @return true Inside
     * @return false Outside
     */
    static bool isInsideRect(const std::vector<cv::Point2f> &rect, const cv::Point2f &pt) {
        int size = rect.size();
        if (size < 4)
            return false;

        cv::Point2f extreme(INFINITY, pt.y);
        int count = 0, i = 0;
        do {
            int next = (i + 1) % size;
            if (isIntersect(rect[i], rect[next], pt, extreme)) {
                if (orientation(rect[i], pt, rect[next]) == 0)
                    return onSegment(rect[i], pt, rect[next]);

                count++;
            }
            i = next;
        } while (i != 0);

        return count & 1;
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
     * @brief Count number of points inside a rectangle
     * with good tracking status (=1)
     * 
     * @param points Vector of points
     * @param corner_pts Vector of corners
     * @param status Vector of status
     * @return int Number of good points
     */
    static int _amountGoodPtInsideRect(std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &corner_pts,
                                       std::vector<unsigned char> &status) {
        CV_Assert(corner_pts.size() == 4);
        CV_Assert(points.size() == status.size());

        // ax+by+c=0
        float a[4];
        float b[4];
        float c[4];

        a[0] = corner_pts[3].y - corner_pts[0].y;
        a[1] = corner_pts[2].y - corner_pts[1].y;
        a[2] = corner_pts[1].y - corner_pts[0].y;
        a[3] = corner_pts[2].y - corner_pts[3].y;

        b[0] = corner_pts[0].x - corner_pts[3].x;
        b[1] = corner_pts[1].x - corner_pts[2].x;
        b[2] = corner_pts[0].x - corner_pts[1].x;
        b[3] = corner_pts[3].x - corner_pts[2].x;

        c[0] = corner_pts[0].y * corner_pts[3].x - corner_pts[3].y * corner_pts[0].x;
        c[1] = corner_pts[1].y * corner_pts[2].x - corner_pts[2].y * corner_pts[1].x;
        c[2] = corner_pts[0].y * corner_pts[1].x - corner_pts[1].y * corner_pts[0].x;
        c[3] = corner_pts[3].y * corner_pts[2].x - corner_pts[2].y * corner_pts[3].x;

        float max_x, min_x, max_y, min_y;
        max_x = corner_pts[0].x;
        min_x = corner_pts[0].x;
        max_y = corner_pts[0].y;
        min_y = corner_pts[0].y;

        int i;
        for (i = 1; i < 4; i++) {
            if (corner_pts[i].x > max_x)
                max_x = corner_pts[i].x;
            if (corner_pts[i].x < min_x)
                min_x = corner_pts[i].x;
            if (corner_pts[i].y > max_y)
                max_y = corner_pts[i].y;
            if (corner_pts[i].y < min_y)
                min_y = corner_pts[i].y;
        }

        float val[4];
        size_t size = points.size();
        int count = 0;
        for (size_t j = 0; j < size; j++) {
            if (status[j] > 0) {
                for (i = 0; i < 4; i++) {
                    val[i] = a[i] * points[j].x + b[i] * points[j].y + c[i];
                }
                if (val[0] * val[1] <= 0 && val[2] * val[3] <= 0) {
                    count++;
                } else {
                    status[j] = 0;
                }
            }
        }

        return count;
    }

    /**
     * @brief Calculates the object coordinates depending on homography matrix
     * 
     * @param pos Position of object
     * @param homo Homography matrix
     * @return std::vector<cv::Point2f> Object coordinates
     */
    static std::vector<cv::Point2f> calcObjPos(const Boundary &pos, cv::Mat &homo) {
        std::vector<cv::Point2f> position;
        if (pos.empty())
            return position;

        cv::Mat src = pointsToMat(pos);
        cv::Mat dst = homo * src;

        cv::Point2f point;
        for (int i = 0; i < dst.cols; ++i) {
            point.x = (float) (dst.at<double>(0, i) / dst.at<double>(2, i));
            point.y = (float) (dst.at<double>(1, i) / dst.at<double>(2, i));
            position.push_back(point);
        }

        return position;
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

    static cv::Mat decodeMat(const std::string &encoded) {
        std::string dec_jpg = base64_decode(encoded);
        std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
        cv::Mat img = cv::imdecode(cv::Mat(data), 0);
        return img;
    }

    static std::string encodeMat(const cv::Mat &img) {
        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf);
        auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
        std::string encoded = base64_encode(enc_msg, buf.size());
        return encoded;
    }

};

#endif // __CVUTILS__