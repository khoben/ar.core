package com.khoben.arcore;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2BGR;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private Mat frame;
    private Mat processedFrame;
    private JavaCameraView javaCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    System.loadLibrary("ar_core-lib");
                    Log.i(TAG, "OpenCV loaded successfully");
                    javaCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
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
        String[] accessPermissions = new String[] {
                Manifest.permission.CAMERA
        };
        boolean needRequire = false;
        for(String access : accessPermissions) {
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
        OpenCVJNI.addMarker(mat_1.getNativeObjAddr());
//        OpenCVJNI.addMarker(mat_2.getNativeObjAddr());
        OpenCVJNI.createMarkerDB();
    }

    public void onPause()
    {
        super.onPause();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.i(TAG, "OpenCV loaded");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
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
        frame = inputFrame.rgba();
        OpenCVJNI.startAR(frame.getNativeObjAddr(), processedFrame.getNativeObjAddr());
        return processedFrame;
    }
}
