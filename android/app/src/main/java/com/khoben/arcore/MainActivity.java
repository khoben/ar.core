package com.khoben.arcore;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    static {
        System.loadLibrary("ar_core-lib");
    }

    private Mat frame;
    private Mat processedFrame;
    private JavaCameraView javaCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    javaCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Asking for permissions
        String[] accessPermissions = new String[]{
                Manifest.permission.CAMERA
        };
        boolean needRequire = false;
        for (String access : accessPermissions) {
            int curPermission = ActivityCompat.checkSelfPermission(this, access);
            if (curPermission != PackageManager.PERMISSION_GRANTED) {
                needRequire = true;
                break;
            }
        }
        if (needRequire) {
            ActivityCompat.requestPermissions(
                    this,
                    accessPermissions,
                    PERMISSION_REQUEST_CODE_CAMERA);
            return;
        }

        javaCameraView = findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(View.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    private void loadMarkers() {
        // load marker
        Mat mat_1 = UtilsJNI.loadMatFromDrawables(this, R.drawable.miku);
        Mat mat_2 = UtilsJNI.loadMatFromDrawables(this, R.drawable.ar);
        Mat mat_3 = UtilsJNI.loadMatFromDrawables(this, R.drawable.czech);
        OpenCVJNI.addMarker(mat_2.getNativeObjAddr());
        OpenCVJNI.addMarker(mat_3.getNativeObjAddr());
        OpenCVJNI.addMarker(mat_1.getNativeObjAddr());
        OpenCVJNI.createMarkerDB();
    }

    public void onPause() {
        super.onPause();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.i(TAG, "started");
        frame = new Mat(height, width, CvType.CV_8UC4);
        processedFrame = new Mat(height, width, CvType.CV_8UC4);
        OpenCVJNI.initAR(frame.getNativeObjAddr());
        loadMarkers();
    }

    @Override
    public void onCameraViewStopped() {
        frame.release();
        processedFrame.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.i(TAG, "on frame");
        frame = inputFrame.rgba();
        OpenCVJNI.startAR(frame.getNativeObjAddr(), processedFrame.getNativeObjAddr());
        return processedFrame;
    }
}
