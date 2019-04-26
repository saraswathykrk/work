package com.projects.posedetector;

import android.app.Activity;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class PoseClassifierPoseNet extends PoseClassifier {

    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
     * of the super class, because we need a primitive array here.
     */
    //private float[][] labelProbArray = null;  //saras

    private static final int HEATMAP_SIZE = 1*TemplateClass.ARR_RESOLUTION*TemplateClass.ARR_RESOLUTION*TemplateClass.num_keypoints;
    private static final int OFFSET_SIZE = 1*TemplateClass.ARR_RESOLUTION*TemplateClass.ARR_RESOLUTION*TemplateClass.offset_res;
    private static final int DISP_FWD_SIZE = 1*TemplateClass.ARR_RESOLUTION*TemplateClass.ARR_RESOLUTION*32;
    private static final int DISP_BWD_SIZE = 1*TemplateClass.ARR_RESOLUTION*TemplateClass.ARR_RESOLUTION*32;
    private static final String TAG = "PoseClassifierPoseNet";

    private float[][][][] heatmap = new float[1][TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][TemplateClass.num_keypoints]; //float[HEATMAP_SIZE]; //
    private float[][][][] offset_2 = new float[1][TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][TemplateClass.offset_res]; //float[HEATMAP_SIZE]; //
    private float[][][][] displacement_fwd_2 = new float[1][TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][32]; //float[HEATMAP_SIZE]; ///
    private float[][][][] displacement_bwd_2 = new float[1][TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][32]; //float[HEATMAP_SIZE]; //

/*    private float[] heatmap = new float[HEATMAP_SIZE]; //
    private float[] offset_2 = new float[OFFSET_SIZE]; //
    private float[] displacement_fwd_2 = new float[DISP_FWD_SIZE]; //
    private float[] displacement_bwd_2 = new float[DISP_BWD_SIZE]; //*/

    Map<Integer, Object > outputs;

    /**
     * Initializes an {@code ImageClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public PoseClassifierPoseNet(Activity activity) throws IOException {
        super(activity);
        //labelProbArray = new float[1][getNumLabels()];
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return TemplateClass.model;
        //return "model-mobilenet_v1_101-16.tflite"; //"quant11_075.tflite"; //"quant12_075.tflite"; //
        //return "model-mobilenet_v1_075.tflite";
        //return "model-mobilenet_v1_101.tflite";
        //return "converted_model1.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "labels_mobilenet_quant_v1_224.txt";
    }

    @Override
    public int getImageSizeX() {
        return TemplateClass.INPUT_SIZE;
    }

    @Override
    public int getImageSizeY() {
        return TemplateClass.INPUT_SIZE;
    }

    @Override
    public int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF) * (2.0f /255) - 1);
        imgData.putFloat(((pixelValue >> 8) & 0xFF) * (2.0f /255) - 1);
        imgData.putFloat((pixelValue & 0xFF) * (2.0f /255) - 1);
     /*   imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
        imgData.put((byte) (pixelValue & 0xFF));*/
    }

    @Override
    protected void runInference() {
        print_debug("POSENET", "in run");
       /* outputs = new HashMap<Integer, Object>() {
            {
                outputs.put(0, heatmap);
                outputs.put(1, offset_2);
                outputs.put(2, displacement_fwd_2);
                outputs.put(3, displacement_bwd_2);
            }};*/

        outputs = new HashMap<>();
        outputs.put(0,heatmap);
        outputs.put(1,offset_2);
        outputs.put(2,displacement_fwd_2);
        outputs.put(3,displacement_bwd_2);

        TemplateClass.poses_detected = false;

        float heatmap_max = heatmap[0][0][0][0];
        float heatmap_min = heatmap[0][0][0][0];
        float off_max = offset_2[0][0][0][0];
        float off_min = offset_2[0][0][0][0];
        float fwd_max = displacement_fwd_2[0][0][0][0];
        float fwd_min = displacement_fwd_2[0][0][0][0];
        float bwd_max = displacement_bwd_2[0][0][0][0];
        float bwd_min = displacement_bwd_2[0][0][0][0];

        for(int i = 0; i<1; i++) {
            for (int j = 0; j < TemplateClass.ARR_RESOLUTION; j++) {
                for (int k = 0; k < TemplateClass.ARR_RESOLUTION; k++) {
                    for (int l = 0; l < TemplateClass.num_keypoints; l++) {
                        //Log.d(TAG, "Heatmap values:" + heatmap[i][j][k][l]);
                        if (heatmap_max < heatmap[i][j][k][l]) {
                            heatmap_max = heatmap[i][j][k][l];
                        }
                        if (heatmap_min > heatmap[i][j][k][l]) {
                            heatmap_min = heatmap[i][j][k][l];
                        }

                        //heatmap_f[i][j][k][l] = (float) heatmap[i][j][k][l];
                    }
                }
            }
        }

        for(int i = 0; i<1; i++) {
            for (int j = 0; j < TemplateClass.ARR_RESOLUTION; j++) {
                for (int k = 0; k < TemplateClass.ARR_RESOLUTION; k++) {
                    for (int l = 0; l < TemplateClass.offset_res; l++) {
                        //Log.d(TAG, "Heatmap values:" + heatmap[i][j][k][l]);
                        if (off_max < offset_2[i][j][k][l]) {
                            off_max = offset_2[i][j][k][l];
                        }
                        if (off_min > offset_2[i][j][k][l]) {
                            off_min = offset_2[i][j][k][l];
                        }

                        //heatmap_f[i][j][k][l] = (float) heatmap[i][j][k][l];
                    }
                }
            }
        }

        for(int i = 0; i<1; i++) {
            for (int j = 0; j < TemplateClass.ARR_RESOLUTION; j++) {
                for (int k = 0; k < TemplateClass.ARR_RESOLUTION; k++) {
                    for (int l = 0; l < 32; l++) {
                        //Log.d(TAG, "Heatmap values:" + heatmap[i][j][k][l]);
                        if (fwd_max < displacement_fwd_2[i][j][k][l]) {
                            fwd_max = displacement_fwd_2[i][j][k][l];
                        }
                        if (fwd_min > displacement_fwd_2[i][j][k][l]) {
                            fwd_min = displacement_fwd_2[i][j][k][l];
                        }

                        //heatmap_f[i][j][k][l] = (float) heatmap[i][j][k][l];
                    }
                }
            }
        }

        for(int i = 0; i<1; i++) {
            for (int j = 0; j < TemplateClass.ARR_RESOLUTION; j++) {
                for (int k = 0; k < TemplateClass.ARR_RESOLUTION; k++) {
                    for (int l = 0; l < 32; l++) {
                        //Log.d(TAG, "Heatmap values:" + heatmap[i][j][k][l]);
                        if (bwd_max < displacement_bwd_2[i][j][k][l]) {
                            bwd_max = displacement_bwd_2[i][j][k][l];
                        }
                        if (bwd_min > displacement_bwd_2[i][j][k][l]) {
                            bwd_min = displacement_bwd_2[i][j][k][l];
                        }

                        //heatmap_f[i][j][k][l] = (float) heatmap[i][j][k][l];
                    }
                }
            }
        }



        Object[] inputs = {imgData};


        //tflite.run(imgData, heatmap); //, offset_2, displacement_fwd_2, displacement_bwd_2);
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        print_debug(TAG, "after run");

        final long FinalStTime = SystemClock.uptimeMillis();

        Log.d(TAG, "Heatmap array value:" + heatmap[0][0][0][0]);
        Log.d(TAG, "Offset array value:"+ offset_2[0][0][0][0]);
        Log.d(TAG, "Disp Fwd array value:" + displacement_fwd_2[0][0][0][0]);
        Log.d(TAG, "Disp Bwd array value:" + displacement_bwd_2[0][0][0][0]);

        MultiPoses multiposes = new MultiPoses();

        TemplateClass.poses_detected = false;

        TemplateClass.multiPoseScores = multiposes.decode_multiple_poses(heatmap[0],
                offset_2[0],
                displacement_fwd_2[0],
                displacement_bwd_2[0],
                TemplateClass.output_stride,
                TemplateClass.max_pose_detections,
                TemplateClass.score_threshold,
                TemplateClass.nms_radius,
                TemplateClass.min_pose_score);

        if (TemplateClass.poses_detected){
            Log.d(TAG, "Heatmap - min/max:" + heatmap_min + "," + heatmap_max);
            Log.d(TAG, "offset - min/max:" + off_min + "," + off_max);
            Log.d(TAG, "disp fwd - min/max:" + fwd_min + "," + fwd_max);
            Log.d(TAG, "disp bwd - min/max:" + bwd_min + "," + bwd_max);
        }

        //TemplateClass.multiPoseScores.pose_key_coords *= output_scale;  //modify saras

        final long finalTime = SystemClock.uptimeMillis() - FinalStTime;  //saras

        Log.d(TAG, "TIME: For detecting poses is:" + finalTime);  //saras

        TemplateClass.poses_detected = true;

        Trace.endSection();

        //return TemplateClass.multiPoseScores;
    }
}
