#include "opencv2/opencv.hpp"
#include "src/AR/Ar.hpp"
#include "src/Utils/CvUtils.hpp"

using namespace cv;

int main(int, char**)
{
//    VideoCapture cap(0); // open the default camera
//    if(!cap.isOpened())  // check if we succeeded
//        return -1;
    AR* ar = new AR();
    cv::Mat mat = cv::imread("C:\\Users\\extle\\Desktop\\OpenCV-Marker-less-AR-master\\WinDemo\\marker\\miku.jpg");
    cv::Mat mat1 = cv::imread("C:\\Users\\extle\\Desktop\\OpenCV-Marker-less-AR-master\\WinDemo\\marker\\czech.jpg");
    ar->add(mat);
    ar->process(mat1);
//    bool res = CvUtils::proveRect(std::vector<cv::Point2f>{
//        cv::Point2f(0,0),
//        cv::Point2f(0, 626),
//        cv::Point2f(449, 626),
//        cv::Point2f(449, 0)
//    });
//
//    std::cout << res <<std::endl;


//    for(;;)
//    {
//        Mat frame;
//        cap >> frame; // get a new frame from camera
//        ar->process(frame);
//        if(waitKey(30) >= 0) break;
//    }

    return 0;
}