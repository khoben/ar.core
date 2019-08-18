#ifndef AR_CORE_ANDROID_HPP
#define AR_CORE_ANDROID_HPP

#include "opencv2/opencv.hpp"
#include "AR/Ar.hpp"
#include "Utils/CvUtils.hpp"

using namespace cv;

#ifdef ANDROID
// LOGS ANDROID
#include <android/log.h>
#define LOG_TAG "AR_CORE_ANDROID"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGSIMPLE(...)
#else
// LOGS NO ANDROID
#include <stdio.h>
#define LOGV(...)        \
    printf("  ");        \
    printf(__VA_ARGS__); \
    printf("\t -  <%s> \n", LOG_TAG);
#define LOGD(...)        \
    printf("  ");        \
    printf(__VA_ARGS__); \
    printf("\t -  <%s> \n", LOG_TAG);
#define LOGI(...)        \
    printf("  ");        \
    printf(__VA_ARGS__); \
    printf("\t -  <%s> \n", LOG_TAG);
#define LOGW(...)            \
    printf("  * Warning: "); \
    printf(__VA_ARGS__);     \
    printf("\t -  <%s> \n", LOG_TAG);
#define LOGE(...)             \
    printf("  *** Error:  "); \
    printf(__VA_ARGS__);      \
    printf("\t -  <%s> \n", LOG_TAG);
#define LOGSIMPLE(...) \
    printf(" ");       \
    printf(__VA_ARGS__);
#endif // ANDROID

/**
 * @brief Make query resised frame
 * 
 * @param size Size of source frame
 * @param max_size Max size
 * @param scale Initial scale
 * @return cv::Mat Query-resized frame
 */
cv::Mat makeQueryMat(cv::Size size, int max_size, int &scale);

/**
 * @brief Init AR instance
 * 
 * @param frame First frame
 */
void init(Mat frame);

/**
 * @brief Add image for tracking purposes
 * 
 * @param img Input image
 * @return int 
 */
int addMarker(cv::Mat img);

/**
 * @brief Create marker 'database'
 * 
 * @return int 
 */
int createMarkerDB();

/**
 * @brief Process frame, find matching in marker 'database'
 * 
 * @param frame Input frame
 * @param out Output frame
 * @return int 
 */
int process(Mat frame, Mat &out);

#endif //AR_CORE_ANDROID_HPP
