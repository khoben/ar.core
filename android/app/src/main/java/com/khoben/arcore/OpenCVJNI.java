package com.khoben.arcore;

public class OpenCVJNI {
    public static native int toGray(long matAddrRgba, long matAddrGray);
    public static native int startAR(long matAddrRgba, long matAddrGray);
    public static native int initAR(long matAddrRgba);
    public static native int addMarker(long matAddrRgba);
    public static native int createMarkerDB();
}
