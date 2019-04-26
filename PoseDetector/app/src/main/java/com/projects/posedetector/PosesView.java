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
    private TemplateClass.multi_pose_scores pose_results;
    private float diff_height;
    private List<Integer[]> adjacent_keypoints;
    private final Paint points;
    private final Paint lines;

    public PosesView(final Context context, final AttributeSet set) {
        super(context, set);

        points = new Paint();
        points.setStrokeWidth(15);
        lines = new Paint();
        lines.setStrokeWidth(15);

        adjacent_keypoints = new ArrayList<>();
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


    @Override
    public void onDraw(final Canvas canvas) {

   /*     Log.d(TAG, "Logo View:" + findViewById(R.id.logoview2).getHeight() + ";" + findViewById(R.id.logoview2).getWidth());
        Log.d(TAG, "Logo View:" + findViewById(R.id.text).getHeight() + ";" + findViewById(R.id.text).getWidth());*/

   Log.d(TAG, "Canvas height:" + getHeight() + "," + diff_height);



        float h = (float) (getHeight() - 0) / INPUT_SIZE;  //need to modify after checking the logoview2 height
        float w = (float) getWidth() / INPUT_SIZE;

        Log.d(TAG, "Canvas size:" + getWidth() + ":" + getHeight());
        // Log.d(TAG, "bitmap size:" + copy.getWidth() + ":" + copy.getHeight());

        if (this.pose_results != null)
        {

            Log.d(TAG,"in here");

            float[] scores = pose_results.scores;
            float[][] pose_key_scores = pose_results.pose_key_scores;
            float[][][] pose_key_coords = pose_results.pose_key_coords;

            for (int i=0; i < scores.length; i++)
            {
                //Log.d(TAG,"in loop 1");

                if (scores[i] < TemplateClass.min_pose_score)
                {
                    //Log.d(TAG,"in loop 11");
                    continue;
                }

                Log.d(TAG, "scores are:" + i + ":" + scores[i]);
                Log.d(TAG, "pose key scores are:" + pose_key_scores[i][0]);
                Log.d(TAG, "pose key coords1 are:" + pose_key_coords[i][0][0]);

                adjacent_keypoints = (get_adjacent_keypoints(pose_key_scores[i], pose_key_coords[i], TemplateClass.min_pose_score, h, w));

                Log.d(TAG,"in loop 2");

                if (i%5==0){
                    //Log.d(TAG,"in loop 20");
                    points.setColor(Color.CYAN);
                    lines.setColor(Color.CYAN);

                }
                else if(i%5==1)
                {
                    //Log.d(TAG,"in loop 21");
                    points.setColor(Color.RED);
                    lines.setColor(Color.RED);
                }
                else if(i%5==2)
                {
                    //Log.d(TAG,"in loop 22");
                    points.setColor(Color.GREEN);
                    lines.setColor(Color.GREEN);
                }
                else if(i%5==3)
                {
                    //Log.d(TAG,"in loop 23");
                    points.setColor(Color.YELLOW);
                    lines.setColor(Color.YELLOW);
                }
                else if(i%5==4)
                {
                    //Log.d(TAG,"in loop 24");
                    points.setColor(Color.MAGENTA);
                    lines.setColor(Color.MAGENTA);
                }

                for (int k=0; k < pose_key_scores[i].length; k++)
                {
                    if (pose_key_scores[i][k] < TemplateClass.min_part_score)
                    {
                        //Log.d(TAG,"in loop 3");
                        continue;
                    }

                    if (k%5==0){
                        //Log.d(TAG,"in loop 20");
                        points.setColor(Color.CYAN);
                        lines.setColor(Color.CYAN);

                    }
                    else if(k%5==1)
                    {
                        //Log.d(TAG,"in loop 21");
                        points.setColor(Color.RED);
                        lines.setColor(Color.RED);
                    }
                    else if(k%5==2)
                    {
                        //Log.d(TAG,"in loop 22");
                        points.setColor(Color.GREEN);
                        lines.setColor(Color.GREEN);
                    }
                    else if(k%5==3)
                    {
                        //Log.d(TAG,"in loop 23");
                        points.setColor(Color.YELLOW);
                        lines.setColor(Color.YELLOW);
                    }
                    else if(k%5==4)
                    {
                        //Log.d(TAG,"in loop 24");
                        points.setColor(Color.MAGENTA);
                        lines.setColor(Color.MAGENTA);
                    }

                    canvas.drawPoint(pose_key_coords[i][k][1] * (w), pose_key_coords[i][k][0] * (h), points);
                    //canvas.drawPoint(pose_key_coords[i][k][1] , pose_key_coords[i][k][0] , points);

                }

                for (int ind = 0; ind < adjacent_keypoints.size(); ind = ind+2)
                {

                    canvas.drawLine(adjacent_keypoints.get(ind)[1],adjacent_keypoints.get(ind)[0],
                            adjacent_keypoints.get(ind+1)[1],adjacent_keypoints.get(ind+1)[0],
                            lines);
                }

            }
        }
    }
}

