#include <jni.h>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "android.hpp"

using namespace std;
using namespace cv;

int toGray(Mat img, Mat& gray);

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_toGray(JNIEnv *env, jclass type, jlong matAddrRgba,
                                                   jlong matAddrGray) {
    Mat& mRgb = *(Mat*)matAddrRgba;
    Mat& mGray = *(Mat*)matAddrGray;

    int conv;
    jint retVal;

    conv = toGray(mRgb, mGray);
    retVal = (jint)conv;

    return retVal;

}

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_initAR(JNIEnv *env, jclass type, jlong matAddrRgba) {
    Mat& mRgb = *(Mat*)matAddrRgba;
    // init AR
    init(mRgb);

    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_startAR(JNIEnv *env, jclass type, jlong matAddrRgba,
                                         jlong matAddrGray) {

    Mat& mRgb = *(Mat*)matAddrRgba;
    Mat& mGray = *(Mat*)matAddrGray;

    int conv;
    jint retVal;

    // proc
    conv = process(mRgb, mGray);
    retVal = (jint)conv;

    return retVal;

}

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_addMarker(JNIEnv *env, jclass type, jlong matAddrRgba) {

    Mat& mRgb = *(Mat*)matAddrRgba;
    addMarker(mRgb);
    return 0;

}

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_createMarkerDB(JNIEnv *env, jclass type) {
    createMarkerDB();
    return 0;
}

int toGray(Mat img, Mat& gray)
{
    cvtColor(img, gray, COLOR_RGBA2GRAY); // Assuming RGBA input

    if (gray.rows == img.rows && gray.cols == img.cols)
    {
        return (1);
    }
    return(0);
}
