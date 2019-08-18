#include "android.hpp"

AR *ar;                       // AR instance
Mat query;                    // query frame
int scale = 1;                // initial scale
const int maxFrameSize = 600; // maximum frame size
bool isTracked = false;       // tracking status: false - not detected
int trackedId = -1;           // id of detected marker

cv::Mat makeQueryMat(cv::Size size, int max_size, int &scale)
{
    int frame_max_size = std::max(size.width, size.height);
    scale = 1;
    while ((frame_max_size / scale) > max_size)
    {
        scale *= 2;
    }
    return cv::Mat(size.height / scale, size.width / scale, CV_8UC1);
}

void init(Mat frame)
{
    // init AR instance
    ar = new AR();
    scale = 1;
    query = makeQueryMat(frame.size(), maxFrameSize, scale);
}

int addMarker(cv::Mat img)
{
    if (ar == nullptr)
        return -1;
    // load marker images
    ar->add(img);
    return 0;
}

int createMarkerDB()
{
    if (ar == nullptr)
        return -1;
    ar->create();
    return 0;
}

int process(Mat frame, Mat &out)
{
    if (ar == nullptr)
        return -1;
    cv::Mat gray;

    if (frame.empty())
        return -1;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    //    frame.copyTo(gray);
    if (isTracked)
    {
        LOGD("CONTINUE..\n");
        isTracked = ar->keepTracking(gray);
    }
    else
    {
        cv::resize(gray, query, query.size());
        std::vector<QueryItem> result = ar->process(query);
        if (!result.empty())
        {
            std::vector<cv::Point2f> objPose;
            QueryItem r = result[0];
            trackedId = r.imgId;
            LOGD("MATCHED: IMG_ID: %d\n", r.imgId);
            std::cout << "Matched: img_id:" << r.imgId << std::endl;
            objPose = r.objPose;
            objPose = CvUtils::scalePoints(objPose, scale);
            std::cout << "Pose: " << objPose << " probability: " << r.probability << std::endl;
            ar->startTracking(gray, objPose);
            isTracked = true;
        }
    }
    frame.copyTo(out);
    if (isTracked)
    {
        cv::Scalar val(255);
        ObjectPosition objPose = ar->getTrackingInstance()->objectPosition;
        cv::line(out, objPose[3], objPose[0], val, 3);
        for (int i = 0; i < 3; i++)
        {
            line(out, objPose[i], objPose[i + 1], val, 3);
        }
        cv::Point center((objPose[0] + objPose[2]) / 2);

        cv::putText(out, std::to_string(trackedId), center, FONT_HERSHEY_TRIPLEX, 6, val, 3);
    }

    return 0;
}
