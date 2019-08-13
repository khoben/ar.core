#include "opencv2/opencv.hpp"
#include "src/AR/Ar.hpp"
#include "src/Utils/CvUtils.hpp"

using namespace cv;

float calcIntegBinDistribution(int in_feats_num, int match_num, float Pp)
{
    float prob = 0;
//	float Np = 1.0 - Pp;
    float tmp1;
    float logPp = log(Pp);
    float logNp = log((float)1.0 - Pp);
    int i,j;

    for(i=0;i<=match_num;i++){
        tmp1 = 0;
        for(j=0;j<i;j++){
            tmp1 += (float)log((double)(in_feats_num - j));
            tmp1 -= (float)log((double)(j+1));
        }
        tmp1 += logPp*i;
        tmp1 += logNp*(in_feats_num-i);
        prob += exp(tmp1);
        if(prob > 1){
            prob = 1;
            break;
        }
    }

    return prob;
}

int main(int, char**)
{
//    VideoCapture cap(0); // open the default camera
//    if(!cap.isOpened())  // check if we succeeded
//        return -1;

    AR* ar = new AR();
    cv::Mat mat = cv::imread(R"(..\resources\marker\miku.jpg)", 0);
    cv::Mat mat_1 = cv::imread(R"(..\resources\marker\czech.jpg)", 0);
    cv::Mat mat_2 = cv::imread("..\\resources\\4.jpg", 0);
    ar->add(std::vector<cv::Mat>{
        mat,
        mat_1
    });
    ar->process(mat_2);


//    cv::Mat grayImg;
//
//    Mat frame;
//    cap >> frame;
//    cv::Size frame_size = frame.size();
//
//    cv::Mat query_image;
//    int frame_max_size;
//    if(frame_size.width > frame_size.height){
//        frame_max_size = frame_size.width;
//    }
//    else{
//        frame_max_size = frame_size.height;
//    }
//    int query_scale = 1;
//    int max_query_size = 320;
//    while((frame_max_size / query_scale) > max_query_size){
//        query_scale*=2;
//    }
//    query_image.create(frame_size.height/query_scale, frame_size.width/query_scale, CV_8UC1);
//
////    ar->process(mat_1);
//
//    for(;;)
//    {
//        cap >> frame; // get a new frame from camera
//        cv::imshow("AR", frame);
//        cv::cvtColor(frame, grayImg, COLOR_BGR2GRAY);
//        cv::resize(grayImg, query_image, query_image.size());
//        ar->process(frame);
//        if(waitKey(30) >= 0) break;
//    }

    return 0;
}