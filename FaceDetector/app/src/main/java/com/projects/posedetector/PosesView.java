package com.projects.posedetector;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import java.util.ArrayList;
import java.util.List;
import static com.projects.posedetector.TemplateClass.INPUT_SIZE;

public class PosesView extends View{

    private static final String TAG = "PosesView";
    private Face_Bbox face_bbox;
    private TemplateClass.multi_pose_scores pose_results;
    private float diff_height;
    private List<Integer[]> adjacent_keypoints;
    private final Paint points;
    private final Paint lines;
    private final Paint rect;
    private class Face_Bbox{
        float bbox_X_nose;
        float bbox_Y_nose;
        float bbox_X_lear;
        float bbox_Y_lear;
        float bbox_X_rear;
        float bbox_Y_rear;
        float hor_dist;
        float vert_dist;
    }

    public PosesView(final Context context, final AttributeSet set) {
        super(context, set);

        points = new Paint();
        points.setStrokeWidth(15);
        lines = new Paint();
        lines.setStrokeWidth(15);
        rect = new Paint();
        rect.setStrokeWidth(10);
        rect.setStyle(Paint.Style.STROKE);
        rect.setColor(Color.CYAN);


        adjacent_keypoints = new ArrayList<>();
        face_bbox = new Face_Bbox();

    }

    public void setPoses(final TemplateClass.multi_pose_scores results, float diff_height)
    {
        this.pose_results = results;
        this.diff_height = diff_height;
        postInvalidate();
    }

    public static List<Integer[]> get_adjacent_keypoints(float[] keypoint_scores,
                                                         float[][] keypoint_coords,
                                                         float min_confidence,
                                                         float scaled_height,
                                                         float scaled_width)
    {
        List<Integer[]> results_array = new ArrayList<>();

        for (int j=0; j < TemplateClass.connected_part_indices.size(); j++)
        {
            int left = TemplateClass.connected_part_indices.get(j)[0];
            int right = TemplateClass.connected_part_indices.get(j)[1];

            if ((keypoint_scores[left] < min_confidence) || (keypoint_scores[right] < min_confidence))
            {
                continue;
            }

            Integer[] left_int = {Math.round(keypoint_coords[left][0] * (scaled_height)), Math.round(keypoint_coords[left][1] * (scaled_width))};
            Integer[] right_int = {Math.round(keypoint_coords[right][0] * (scaled_height)), Math.round(keypoint_coords[right][1] * (scaled_width))};

            results_array.add(left_int);
            results_array.add(right_int);
        }
        Log.d(TAG, "results are:" + results_array.size());
        return results_array;
    }

    public Face_Bbox get_bounding_box(float[] keypoint_scores,
                                      float[][] keypoint_coords,
                                      float min_confidence,
                                      float scaled_height,
                                      float scaled_width)
    {
        Face_Bbox bbox = new Face_Bbox(){};

        for (int j=0; j < TemplateClass.parts_ids.size(); j++)
        {
            Log.d(TAG, "BB j values are:" + j);
            Log.d(TAG, "BB keypoint_scores are:" + keypoint_scores[j]);
            for (int i=0; i < keypoint_coords[j].length; i++) {
                Log.d(TAG, "BB i values are:" + i);
                Log.d(TAG, "BB keypoint_coords are:" + keypoint_coords[j][i]);


                if (j == 0) {
                    bbox.bbox_X_nose = keypoint_coords[j][1] * (scaled_width);
                    bbox.bbox_Y_nose = keypoint_coords[j][0] * (scaled_height);
                }
                if (j == 3) {
                    bbox.bbox_X_lear = keypoint_coords[j][1] * (scaled_width);
                    bbox.bbox_Y_lear  = keypoint_coords[j][0] * (scaled_height);
                }
                if (j == 4) {
                    bbox.bbox_X_rear = keypoint_coords[j][1] * (scaled_width);
                    bbox.bbox_Y_rear = keypoint_coords[j][0] * (scaled_height);
                }
            }
        }
//        Log.d(TAG, "BB values:" + face_bbox.bbox_X_lear + "," + face_bbox.bbox_Y_lear + "," + face_bbox.bbox_X_nose + "," + face_bbox.bbox_Y_nose);
        if ((bbox.bbox_X_lear != 0.0f) && (bbox.bbox_X_rear != 0.0f) &&
                (bbox.bbox_X_nose != 0.0f) && (bbox.bbox_Y_nose != 0.0f) &&
                (bbox.bbox_Y_lear != 0.0f) && (bbox.bbox_Y_rear != 0.0f)) {
           // bbox.hor_dist = Math.max(bbox.bbox_X_lear, bbox.bbox_X_nose) - Math.min(bbox.bbox_X_lear, bbox.bbox_X_nose);
           // bbox.vert_dist = Math.max(bbox.bbox_Y_lear, bbox.bbox_Y_nose) - Math.min(bbox.bbox_Y_lear, bbox.bbox_Y_nose);


            float x = (float) (Math.pow(bbox.bbox_X_lear, 2) - Math.pow(bbox.bbox_X_nose,2));
            float y = (float) (Math.pow(bbox.bbox_Y_lear, 2) - Math.pow(bbox.bbox_Y_nose,2));
            bbox.hor_dist = (float) Math.pow(x+y, 0.5);
            Log.d(TAG, "BB values:" + bbox.bbox_X_lear + "," + bbox.bbox_Y_lear + "," + bbox.bbox_X_nose + "," + bbox.bbox_Y_nose + "," +
                    bbox.bbox_X_rear + "," + bbox.bbox_Y_rear);
            return bbox;
        }
        else
            return null;
    }

    @Override
    public void onDraw(final Canvas canvas) {

        Log.d(TAG, "Canvas height:" + getHeight() + "," + diff_height);

        float h = (float) getHeight() / INPUT_SIZE;
        float w = (float) getWidth() / INPUT_SIZE;

        Log.d(TAG, "Canvas size:" + getWidth() + ":" + getHeight());

        if (this.pose_results != null)
        {

            Log.d(TAG,"in here");

            float[] scores = pose_results.scores;
            float[][] pose_key_scores = pose_results.pose_key_scores;
            float[][][] pose_key_coords = pose_results.pose_key_coords;

            for (int i=0; i < scores.length; i++)
            {
                if (scores[i] < TemplateClass.min_pose_score)
                {
                    continue;
                }

                Log.d(TAG, "scores are:" + i + ":" + scores[i]);
                Log.d(TAG, "pose key scores are:" + pose_key_scores[i][0]);
                Log.d(TAG, "pose key coords1 are:" + pose_key_coords[i][0][0]);

                adjacent_keypoints = (get_adjacent_keypoints(pose_key_scores[i], pose_key_coords[i], TemplateClass.min_pose_score, h, w));
                face_bbox = get_bounding_box(pose_key_scores[i], pose_key_coords[i], TemplateClass.min_pose_score, h, w);

                Log.d(TAG,"in loop 2");

                if (i%5==0){
                    points.setColor(Color.CYAN);
                    lines.setColor(Color.CYAN);

                }
                else if(i%5==1)
                {
                    points.setColor(Color.RED);
                    lines.setColor(Color.RED);
                }
                else if(i%5==2)
                {
                    points.setColor(Color.GREEN);
                    lines.setColor(Color.GREEN);
                }
                else if(i%5==3)
                {
                    points.setColor(Color.YELLOW);
                    lines.setColor(Color.YELLOW);
                }
                else if(i%5==4)
                {
                    points.setColor(Color.MAGENTA);
                    lines.setColor(Color.MAGENTA);
                }

                for (int k=0; k < pose_key_scores[i].length; k++)
                {
                    if (pose_key_scores[i][k] < TemplateClass.min_part_score)
                    {
                        continue;
                    }

                    if (k%5==0){
                        points.setColor(Color.BLACK);
                        lines.setColor(Color.BLACK);

                    }
                    else if(k%5==1)
                    {
                        points.setColor(Color.DKGRAY);
                        lines.setColor(Color.DKGRAY);
                    }
                    else if(k%5==2)
                    {
                        points.setColor(Color.BLUE);
                        lines.setColor(Color.BLUE);
                    }
                    else if(k%5==3)
                    {
                        points.setColor(Color.WHITE);
                        lines.setColor(Color.WHITE);
                    }
                    else if(k%5==4)
                    {
                        points.setColor(Color.LTGRAY);
                        lines.setColor(Color.LTGRAY);
                    }

                    points.setColor(Color.CYAN);

                    canvas.drawPoint(pose_key_coords[i][k][1] * (w), pose_key_coords[i][k][0] * (h), points);

                }

                for (int ind = 0; ind < adjacent_keypoints.size(); ind = ind+2)
                {

                    lines.setColor(Color.CYAN);
                    canvas.drawLine(adjacent_keypoints.get(ind)[1],adjacent_keypoints.get(ind)[0],
                            adjacent_keypoints.get(ind+1)[1],adjacent_keypoints.get(ind+1)[0],
                            lines);
                }

                //canvas.drawLine(face_bbox.bbox_X_lear, face_bbox.bbox_Y_lear, face_bbox.bbox_X_nose, face_bbox.bbox_Y_nose, lines);
                //canvas.drawLine(face_bbox.bbox_X_lear, face_bbox.bbox_Y_lear, face_bbox.bbox_X_nose, face_bbox.bbox_Y_nose, lines);
                if (face_bbox != null) {
                    float rect_start_X = face_bbox.bbox_X_rear;
                    float rect_start_Y = (face_bbox.bbox_Y_rear - (face_bbox.hor_dist * 1.0f));
                    float rect_stop_X = face_bbox.bbox_X_lear;
                    float rect_stop_Y = (face_bbox.bbox_Y_lear + (face_bbox.hor_dist * 1.0f));
                    canvas.drawRect(rect_start_X, rect_start_Y, rect_stop_X, rect_stop_Y, rect);
                    //canvas.drawLine(face_bbox.bbox_X_rear, face_bbox.bbox_Y_rear, face_bbox.bbox_X_rear, face_bbox.bbox_Y_rear + face_bbox.hor_dist, lines);
                }

            }
        }
    }
}

