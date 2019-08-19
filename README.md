# Mobile/PC Markerless AR tracker app

C++/CMake core project for **Markerless AR** detector&tracker app powered by OpenCV.

Currently have PC and Android version (JNI).

### Based on [this project](https://github.com/takmin/OpenCV-Marker-less-AR)

## Built With

* CMake 3.6
* OpenCV 4.1 (PC and Android)
* Android NDK 20

## Test

### Markers
![img1](https://github.com/khoben/ar.core/blob/master/README.md-images/czech.jpg)
![img2](https://github.com/khoben/ar.core/blob/master/README.md-images/miku.jpg)

### Recognition
![img3](https://github.com/khoben/ar.core/blob/master/README.md-images/2.png)
![img4](https://github.com/khoben/ar.core/blob/master/README.md-images/1.png)

### Known issues
* Unable to find multiple objects
* Bad recognition when too few features have been extracted from marker`s image


