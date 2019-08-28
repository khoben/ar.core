#include <jni.h>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "android.hpp"

using namespace std;
using namespace cv;

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_init(JNIEnv *env, jclass type, jlong matAddrRgba) {
    Mat& mRgb = *(Mat*)matAddrRgba;
    // init AR
    init(mRgb);

    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_khoben_arcore_OpenCVJNI_start(JNIEnv *env, jclass type, jlong matAddrRgba,
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
