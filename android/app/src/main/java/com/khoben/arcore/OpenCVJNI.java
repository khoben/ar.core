package com.khoben.arcore;

public class OpenCVJNI {
    public static native int start(long matAddrRgba, long matAddrGray);
    public static native int init(long matAddrRgba);
    public static native int addMarker(long matAddrRgba);
}
