#include "opencv2/opencv.hpp"
#include "src/AR/Ar.hpp"

using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    AR* ar = new AR();

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        ar->processFrame(frame);
        if(waitKey(30) >= 0) break;
    }

    return 0;
}