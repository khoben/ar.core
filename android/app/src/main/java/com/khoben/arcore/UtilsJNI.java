package com.khoben.arcore;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.io.InputStream;

public class UtilsJNI {
    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }

        return bitmap;
    }

    public static Mat loadMatFromDrawables(Context context, int idDrawable)
    {
        Mat mat = null;
        try {
            mat = Utils.loadResource(context, idDrawable, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
//            Imgproc.cvtColor(mat, mat, COLOR_RGB2BGR);
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert mat != null;
        return mat;
    }
}
