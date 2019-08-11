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

class CvUtils
{
public:
    /**
     * @brief Create a Mask object
     * 
     * @param size - size of frame
     * @param pose - pose of traking object
     * @return cv::Mat - Mask object
     */
    static cv::Mat createMask(cv::Size size, const ObjectPosition& pose)
    {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        cv::fillConvexPoly(mask, convertVecType<cv::Point>(pose), cv::Scalar(255), 16, 0);
        return mask;
    }

    static std::vector<cv::Point2f> calcObjPos(const ObjectPosition& pos, cv::Mat &homo) {
        std::vector<cv::Point2f> position;
        if (pos.empty())
            return position;

        cv::Mat src = pointsToMat(pos);
        cv::Mat dst = homo * src;

        cv::Point2f point;
        for (int i = 0; i < dst.cols; ++i) {
            point.x = (float) (dst.at<double>(0,i) / dst.at<double>(2,i));
            point.y = (float) (dst.at<double>(1,i) / dst.at<double>(2,i));
            position.push_back(point);
        }

        return position;
    }

    static cv::Mat pointsToMat(const ObjectPosition& v)
    {
        int size = v.size();
        cv::Mat ret(3, size, CV_64FC1);

        for (int i = 0; i < size; ++i) {
            ret.at<double>(0, i) = (double)v[i].x;
            ret.at<double>(1, i) = (double)v[i].y;
            ret.at<double>(2, i) = 1.;
        }
        return ret;
    }

    template <typename T, typename F>
    static std::vector<T> convertVecType(const std::vector<F> &v)
    {
        std::vector<T> ret;
        std::transform(v.begin(), v.end(),
                       std::back_inserter(ret),
                       [](const F& p) { return (T)p; });
        return ret;
    }

    static bool ptsInsideFrame(const cv::Size& size, const std::vector<cv::Point2f>& pts)
    {
        return pts.end() != std::find_if(pts.begin(), pts.end(),
                                         [size](const cv::Point2f &p) -> bool
                                         {
                                             return p.x < 0 or p.x > size.width or p.y < 0 or p.y > size.height;
                                         });
    }

    static bool isOrthogonal(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c)
    {
        return ((b.x - a.x) * (b.x - c.x) +
                (b.y - a.y) * (b.y - c.y)) == 0;
    }

    static bool isRect(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d)
    {
        return isOrthogonal(a, b, c) and
                isOrthogonal(b, c, d) and
                isOrthogonal(c, d, a);
    }

    static bool isRectAnyOrder(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d)
    {
        return isRect(a, b, c, d) ||
               isRect(b, c, a, d) ||
               isRect(c, a, b, d);
    }

    static bool proveRect(const std::vector<cv::Point2f>& pts)
    {
        if (pts.size()!=4)
            return false;

        return isRectAnyOrder(pts[0], pts[1], pts[2], pts[3]);
    }

    static int
    amountGoodPtInsideRect(const std::vector<cv::Point2f>& pts, const std::vector<cv::Point2f>& corners, const std::vector<uchar>& status) {
        if (pts.size()!=status.size() or corners.size()!=4)
            return -1;

        int count = 0;


        return count;
    }
};

#endif // __CVUTILS__