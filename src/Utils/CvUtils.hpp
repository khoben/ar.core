#pragma once

#ifndef __CVUTILS__
#define __CVUTILS__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

// [0]:Top-Left
// [1]:Bottom-Left
// [2]:Bottom-Right
// [3]:Top-Right
typedef std::vector<cv::Point2f> ObjectPosition;

class CvUtils {
public:
    /**
     * @brief Create a Mask object
     * 
     * @param size - size of frame
     * @param pose - pose of traking object
     * @return cv::Mat - Mask object
     */
    static cv::Mat createMask(cv::Size size, const ObjectPosition &pose) {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        cv::fillConvexPoly(mask, convertVecType<cv::Point>(pose), cv::Scalar(255), 16, 0);
        return mask;
    }

    static std::vector<cv::Point2f> calcObjPos(const ObjectPosition &pos, cv::Mat &homo) {
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

    static cv::Mat pointsToMat(const ObjectPosition &v) {
        int size = v.size();
        cv::Mat ret(3, size, CV_64FC1);

        for (int i = 0; i < size; ++i) {
            ret.at<double>(0, i) = (double) v[i].x;
            ret.at<double>(1, i) = (double) v[i].y;
            ret.at<double>(2, i) = 1.;
        }
        return ret;
    }

    template<typename T, typename F>
    static std::vector<T> convertVecType(const std::vector<F> &v) {
        std::vector<T> ret;
        std::transform(v.begin(), v.end(),
                       std::back_inserter(ret),
                       [](const F &p) { return (T) p; });
        return ret;
    }

    static bool ptsInsideFrame(const cv::Size &size, const std::vector<cv::Point2f> &pts) {
        return pts.end() != std::find_if(pts.begin(), pts.end(),
                                         [size](const cv::Point2f &p) -> bool {
                                             return p.x < 0 or p.x > size.width or p.y < 0 or p.y > size.height;
                                         });
    }

    static bool isOrthogonal(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c) {
        return ((b.x - a.x) * (b.x - c.x) +
                (b.y - a.y) * (b.y - c.y)) == 0;
    }

    static bool isRect(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c, const cv::Point2f &d) {
        return isOrthogonal(a, b, c) and
               isOrthogonal(b, c, d) and
               isOrthogonal(c, d, a);
    }

    static bool isRectAnyOrder(const cv::Point2f &a, const cv::Point2f &b, const cv::Point2f &c, const cv::Point2f &d) {
        return isRect(a, b, c, d) or
               isRect(b, c, a, d) or
               isRect(c, a, b, d);
    }

    static bool proveRect(std::vector<cv::Point2f> &pts) {
        if (pts.size() != 4)
            return false;

        const float eps = 1e-5;
        for(cv::Point2f &i: pts){
            if (fabs(i.x) < eps)
                i.x = 0;
            if (fabs(i.y) < eps)
                i.y = 0;
        }

        return isRectAnyOrder(pts[0], pts[1], pts[2], pts[3]);
    }

    static bool _proveRect(std::vector<cv::Point2f>& rect_pts)
    {
        CV_Assert(rect_pts.size()==4);

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
        if(val > 0)
            s = 1;
        else
            s = -1;

//	if(vec[3][0] * vec[0][0] + vec[3][1] * vec[0][1] >= 0)
//		result_f = false;

        for(i=0; i<3; i++){
//		if(vec[i][0] * vec[i+1][0] + vec[i][1] * vec[i+1][1] >= 0){
//			result_f = false;
//			break;
//		}
            val = vec[i][0] * vec[i+1][1] - vec[i][1] * vec[i+1][0];
            if( val * s <= 0){
                result_f = false;
                break;
            }
        }

        return result_f;
    }

    // Given three colinear points p, q, r, the function checks if
    // point q lies on line segment 'pr'
    static bool onSegment(const cv::Point2f &p, const cv::Point2f &q, const cv::Point2f &r) {
        return q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) and
               q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y);
    }

    // To find orientation of ordered triplet (p, q, r).
    // The function returns following values
    // 0 --> p, q and r are colinear
    // 1 --> Clockwise
    // 2 --> Counterclockwise
    static int orientation(const cv::Point2f &p, const cv::Point2f &q, const cv::Point2f &r) {
        int val = (q.y - p.y) * (r.x - q.x) -
                  (q.x - p.x) * (r.y - q.y);

        if (val == 0) return 0;  // colinear
        return (val > 0) ? 1 : 2; // clock or counterclock wise
    }

    // The function that returns true if line segment 'p1q1'
    // and 'p2q2' intersect.
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
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;

        // p1, q1 and p2 are colinear and q2 lies on segment p1q1
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;

        // p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;

        // p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;

        return false; // Doesn't fall in any of the above cases
    }

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

    static int
    amountGoodPtInsideRect(const std::vector<cv::Point2f> &pts, const std::vector<cv::Point2f> &corners,
                           std::vector<uchar> &status) {
        if (pts.size() != status.size() or corners.size() != 4)
            return -1;

        int count = 0, idx = 0;
        std::for_each(pts.begin(), pts.end(), [&idx, &count, &corners, &status](const cv::Point2f &pt) {
            if (status[idx] == 1) {
                bool isInside = isInsideRect(corners, pt);
                if (isInside) count++;
                else status[idx] = 0;
            }
            idx++;
        });

        return count;
    }

    static std::vector<cv::Point2f> affineTransformRect(cv::Size size, cv::Mat mat) {
        std::vector<cv::Point2f> points;
        float width = (float) (size.width) - 1;
        float height = (float) (size.height) - 1;
        cv::Mat srcMat = (cv::Mat_<double>(3, 4) << 0, 0, width, width, 0, height, height, 0, 1, 1, 1, 1);
        cv::Mat destMat = mat * srcMat;
        cv::Point2f pt;
        for (int i = 0; i < 4; i++) {
            pt.x = (float) (destMat.at<double>(0, i) / destMat.at<double>(2, i));
            pt.y = (float) (destMat.at<double>(1, i) / destMat.at<double>(2, i));
            points.push_back(pt);
        }

        return points;
    }
};

#endif // __CVUTILS__