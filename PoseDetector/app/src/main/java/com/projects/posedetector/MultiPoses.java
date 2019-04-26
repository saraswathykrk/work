package com.projects.posedetector;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.util.Log;
import android.view.View;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import com.projects.posedetector.TemplateClass.*;

import static com.projects.posedetector.PoseClassifier.print_debug;
import static com.projects.posedetector.TemplateClass.local_maximum_radius;
import static com.projects.posedetector.TemplateClass.num_edges;
import static com.projects.posedetector.TemplateClass.num_keypoints;
import static com.projects.posedetector.TemplateClass.num_parts;
import static com.projects.posedetector.TemplateClass.part_names;
import static com.projects.posedetector.TemplateClass.squared_nms_radius;

class MultiPoses {
    private static final String TAG = "MultiPoses";

    //protected static final boolean SAVE_PREVIEW_BITMAP = true;

    private ArrayList<scored_parts_data> build_part_with_score_fast(float score_threshold,
                                                                    float[][][] heatscores)
    {
        ArrayList<array_pos> parts_pos = new ArrayList<> ();
        ArrayList<scored_parts_data> parts = new ArrayList<> ();

        float[][] kp_scores = new float[TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION];
        float[][] max_vals = new float[TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION];
        boolean[][] max_loc = new boolean[TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION];

        int score_0 = heatscores.length;
        int score_1 = heatscores[score_0 - 1].length;
        int num_keypoints = heatscores[score_0 - 1][score_1 - 1].length;

        int count_kp, tot_count_kp, count_max, count_loc = 0, parts_count = 0;
        float maximum_val;

        for (int k = 0; k < num_keypoints; k++)
        {
            count_kp = 0;
            tot_count_kp = 0;
            count_max = 0;

            for (int i = 0; i < score_0; i++)
            {
                for (int j = 0; j < score_1; j++)
                {
                    if (heatscores[i][j][k] < score_threshold)
                    {
                        kp_scores[i][j] = 0.0f;
                    } else
                    {
                        kp_scores[i][j] = heatscores[i][j][k];
                        print_debug(TAG, "inside multiposes::" + heatscores[i][j][k]);
                        print_debug(TAG, "new1: kp_scores:" + kp_scores[i][j] + "; for more" + heatscores[i][j][k]);
                        count_kp++;
                    }
                    tot_count_kp++;
                }
            }

            print_debug(TAG, "kp_scores: count :" + count_kp + "keypoint_id:" + k + "size:" + tot_count_kp);
            for (int i = 0; i < score_0; i++)
            {
                for (int j = 0; j < score_1; j++)
                {
                    maximum_val = 0.0f;
                    for (int filter_i = i-local_maximum_radius; filter_i <= i+local_maximum_radius; filter_i++)
                    {
                        for (int filter_j = j-local_maximum_radius; filter_j <= j+local_maximum_radius; filter_j++)
                        {
                            if (!((filter_i < 0) || (filter_i >= score_0)
                                    || (filter_j < 0) || (filter_j >= score_1)))
                            {
                                if (maximum_val < kp_scores[filter_i][filter_j])
                                {
                                    maximum_val = kp_scores[filter_i][filter_j];
                                }
                            }
                        }
                    }
                    max_vals[i][j] = maximum_val;
                    count_max++;
                }
            }

            print_debug(TAG, "max_vals: count :" + count_max);

            for (int i = 0; i < score_0; i++)
            {
                for (int j = 0; j < score_1; j++)
                {
                    max_loc[i][j] = ((kp_scores[i][j] == max_vals[i][j]) && (kp_scores[i][j] > 0.0f));
                    if (max_loc[i][j])
                    {
                        parts_pos.add(new array_pos(i, j));  //adding y first
                        parts.add(new scored_parts_data(heatscores[parts_pos.get(count_loc).x_pos][parts_pos.get(count_loc).y_pos][k],
                                k,
                                parts_pos.get(count_loc).x_pos,
                                parts_pos.get(count_loc).y_pos));
                        print_debug(TAG, "new1: parts value:" + parts.get(parts_count).scores + ":" + parts.get(parts_count).keypoint_id + ":" +
                                parts.get(parts_count).arr_x + ":" + parts.get(parts_count).arr_y);
                        count_loc++;
                        parts_count++;
                    }
                }
            }
            print_debug(TAG, "max_loc: count :" + count_loc + "parts_pos.size" + parts_pos.size());
        }
        print_debug(TAG, "new1: parts count:" + parts_count + "parts.size:" + parts.size());
        return parts;
    }

    private targets traverse_to_targ_keypoint(int edge_id,
                                              float[] source_keypoint,
                                              int target_keypoint_id,
                                              float[][][] heatscores,
                                              float[][][][] offsetArr,
                                              float output_stride,
                                              float[][][][] dispArr)
    {
        float score = 0.0f;
        int height = heatscores.length;
        int width = heatscores[height-1].length;
        float[] image_coord = new float[2];
        int[] source_keypoint_indices = new int[2];
        float[] displaced_point = new float[2];
        int[] displaced_point_indices = new int[2];
        targets new_targets = new targets(score, image_coord);

        source_keypoint_indices[0] = Math.round((source_keypoint[0]/output_stride));
        source_keypoint_indices[1] = Math.round((source_keypoint[1]/output_stride));

        if (source_keypoint_indices[0] < 0){
            source_keypoint_indices[0] = 0;
        }
        else if (source_keypoint_indices[0] > (height - 1)){
            source_keypoint_indices[0] = height - 1;
        }

        if (source_keypoint_indices[1] < 0){
            source_keypoint_indices[1] = 0;
        }
        else if (source_keypoint_indices[1] > (width - 1)){
            source_keypoint_indices[1] = width - 1;
        }

        displaced_point[0] = source_keypoint[0] + dispArr[source_keypoint_indices[0]][source_keypoint_indices[1]][edge_id][0];
        displaced_point[1] = source_keypoint[1] + dispArr[source_keypoint_indices[0]][source_keypoint_indices[1]][edge_id][1];

        displaced_point_indices[0] = Math.round((displaced_point[0]/output_stride));
        displaced_point_indices[1] = Math.round((displaced_point[1]/output_stride));

        if (displaced_point_indices[0] < 0){
            displaced_point_indices[0] = 0;
        }
        else if (displaced_point_indices[0] > (height - 1)){
            displaced_point_indices[0] = height - 1;
        }

        if (displaced_point_indices[1] < 0){
            displaced_point_indices[1] = 0;
        }
        else if (displaced_point_indices[1] > (width - 1)){
            displaced_point_indices[1] = width - 1;
        }

        score = heatscores[displaced_point_indices[0]][displaced_point_indices[1]][target_keypoint_id];
        image_coord[0] = (displaced_point_indices[0] * output_stride) + offsetArr[displaced_point_indices[0]][displaced_point_indices[1]][target_keypoint_id][0];
        image_coord[1] = (displaced_point_indices[1] * output_stride) + offsetArr[displaced_point_indices[0]][displaced_point_indices[1]][target_keypoint_id][1];

        print_debug(TAG, "new2: traverse score" + score);
        new_targets.scores = score;
        new_targets.image_coord = image_coord;  //note that y is first
        return new_targets;
    }

    private targets_array decode_pose(float root_score,
                                      int root_id,
                                      float root_coord_x,
                                      float root_coord_y,
                                      float[][][] heatscores,
                                      float[][][][] offsetArr,
                                      float output_stride,
                                      float[][][][] dispFwdArr,
                                      float[][][][] dispBwdArr)
    {

        int target_keypoint_id;
        int source_keypoint_id;
        ArrayList<targets> pose_targets = new ArrayList<> ();
        targets_array instance_targets;

        int pose_ind = 0;
        float[] instance_keypoint_scores = new float[num_parts];
        float[][] instance_keypoint_coords = new float[num_parts][2];
        instance_keypoint_scores[root_id] = root_score;
        instance_keypoint_coords[root_id][0] = root_coord_x;
        instance_keypoint_coords[root_id][1] = root_coord_y;

        for(int edge=15; edge>=0; edge--)
        {
            target_keypoint_id = TemplateClass.parent_child_tuples.get(edge)[0];
            source_keypoint_id = TemplateClass.parent_child_tuples.get(edge)[1];

            if (instance_keypoint_scores[source_keypoint_id] > 0.0 & instance_keypoint_scores[target_keypoint_id] == 0.0)
            {
                pose_targets.add(traverse_to_targ_keypoint(edge,
                        instance_keypoint_coords[source_keypoint_id],
                        target_keypoint_id,
                        heatscores, offsetArr, output_stride, dispBwdArr));
                instance_keypoint_scores[target_keypoint_id] = pose_targets.get(pose_ind).scores;
                instance_keypoint_coords[target_keypoint_id] = pose_targets.get(pose_ind).image_coord;
                print_debug(TAG, "new3: bwd pose_targets.get(pose_ind).scores:" + pose_targets.get(pose_ind).scores);
                pose_ind++;
            }
        }

        for(int edge=0; edge < num_edges; edge++)
        {
            source_keypoint_id = TemplateClass.parent_child_tuples.get(edge)[0];
            target_keypoint_id = TemplateClass.parent_child_tuples.get(edge)[1];

            if (instance_keypoint_scores[source_keypoint_id] > 0.0 & instance_keypoint_scores[target_keypoint_id] == 0.0)
            {
                pose_targets.add(traverse_to_targ_keypoint(edge,
                        instance_keypoint_coords[source_keypoint_id],
                        target_keypoint_id,
                        heatscores, offsetArr, output_stride, dispFwdArr));
                instance_keypoint_scores[target_keypoint_id] = pose_targets.get(pose_ind).scores;
                instance_keypoint_coords[target_keypoint_id] = pose_targets.get(pose_ind).image_coord;
                print_debug(TAG, "new3: fwd pose_targets.get(pose_ind).scores:" + pose_targets.get(pose_ind).scores);
                pose_ind++;
            }
        }

        print_debug(TAG, "new5: instance_keypoint_scores:" + instance_keypoint_scores[0]);
        instance_targets = new targets_array(instance_keypoint_scores, instance_keypoint_coords);
        for(int i=0; i<instance_targets.scores.length; i++)
        {
            print_debug(TAG, "new5: instance_targets scores:" + instance_targets.scores[i]);
        }
        return instance_targets;
    }


    private float get_instance_score_fast(float[][][] exist_pose_coords,
                                          float[] keypoint_scores,
                                          float[][] keypoint_coords) {
        float[][][] diff;
        float[][][] new_diff;
        float[][] out;
        boolean[][] s;
        boolean[] s_new;
        float not_overlapped_scores = 0.0f;
        int h = exist_pose_coords.length;
        print_debug(TAG, "new6: not length:" + h);

        if (h > 0)
        {
            int w = exist_pose_coords[h-1].length;
            diff = new float[h][w][2];
            new_diff = new float[h][w][2];
            out = new float[h][w];
            s = new boolean[h][w];
            s_new = new boolean[w];

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    diff[i][j][0] = (exist_pose_coords[i][j][0] - keypoint_coords[j][0]);
                    diff[i][j][1] = (exist_pose_coords[i][j][1] - keypoint_coords[j][1]);
                }
            }

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    new_diff[i][j][0] = diff[i][j][0] * diff[i][j][0];
                    new_diff[i][j][1] = diff[i][j][1] * diff[i][j][1];
                    print_debug(TAG, "new6: not diff:" + new_diff[i][j][0] + ":" + new_diff[i][j][1]);
                }
            }

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    out[i][j] = new_diff[i][j][0] + new_diff[i][j][1];
                    s[i][j] = (out[i][j] > squared_nms_radius);
                }
            }

            for (int j = 0; j < w; j++)
            {
                s_new[j] = true;
                for (int i = 0; i < h; i++)
                {
                    s_new[j] = s_new[j] && s[i][j];
                }
                if (s_new[j])
                {
                    not_overlapped_scores = not_overlapped_scores + keypoint_scores[j];
                }
            }

            /*for(int j=0; j < w; j++)
            {
                if (s_new[j])
                {
                    not_overlapped_scores = not_overlapped_scores + keypoint_scores[j];
                }
            }*/
            print_debug(TAG, "new6: not_overlapped_scores in if:" + not_overlapped_scores);
        }
        else
        {
            for(int i=0; i < keypoint_scores.length; i++)
            {
                not_overlapped_scores = not_overlapped_scores + keypoint_scores[i];
            }
            print_debug(TAG, "new6: not_overlapped_scores in else:" + not_overlapped_scores);
        }

        print_debug(TAG, "new6: pose_score in func:" + not_overlapped_scores/keypoint_scores.length);

        return not_overlapped_scores/keypoint_scores.length;

    }


    private boolean within_nms_radius_fast(float[][] pose_coords,
                                           float[] point)
    {
        float[][] new_diff;
        float[] out;
        boolean s = false;

        int h = pose_coords.length;

        print_debug(TAG, "new7: pose_coords.length::" + h);

        if (pose_coords.length == 0)
        {
            print_debug(TAG, "new7: returning false");
            return false;
        }
        else
        {
            new_diff = new float[pose_coords.length][2];
            out = new float[pose_coords.length];
            for(int i = 0; i < pose_coords.length; i++)
            {
                print_debug(TAG, "new7: pose_coords.length" + pose_coords.length);
                print_debug(TAG, "new7: pose_coords" + pose_coords[i][0] + "," + pose_coords[i][1]);
                new_diff[i][0] = pose_coords[i][0] - point[0];
                new_diff[i][1] = pose_coords[i][1] - point[1];
                out[i] = (new_diff[i][0] * new_diff[i][0]) + (new_diff[i][1] * new_diff[i][1]);  //modify saras check once

                print_debug(TAG, "new7: new_diff" + new_diff[i][0] + "," + new_diff[i][1]);
            }

            for(int i = 0; i < pose_coords.length; i++)
            {
                s = (out[i] <= squared_nms_radius);

                print_debug(TAG, "new7: out" + out[i]);
                if (s)
                {
                    break;
                }
            }

            print_debug(TAG, "new7: returning s:" + s);

            return s;
        }
    }

    public multi_pose_scores decode_multiple_poses(float[][][] heatmapArr,
                                                   float[][][] offsetArr,
                                                   float[][][] dispFwdArr,
                                                   float[][][] dispBwdArr,
                                                   int output_stride,
                                                   int max_pose_detections,
                                                   float score_threshold,
                                                   float nms_radius,
                                                   float min_pose_score)
    {
        int pose_count = 0;
        float root_score;
        int root_id;
        int root_coord_x;
        int root_coord_y;
        float[] root_image_coords;
        int count_o;
        int k_offset;
        int count_f;
        int k_disp_f;
        int count_b;
        int k_disp_b;
        float pose_score;
        int height;
        int width;

        float[][][][] offsetNewArr = new float[TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][TemplateClass.num_keypoints][2];  //modify saras this hard-coding if possible
        float[][][][] dispNewFwdArr = new float[TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][16][2];
        float[][][][] dispNewBwdArr = new float[TemplateClass.ARR_RESOLUTION][TemplateClass.ARR_RESOLUTION][16][2];

        float[] pose_scores = new float[max_pose_detections];
        float[][] pose_keypoint_scores = new float[max_pose_detections][num_keypoints];
        float[][][] pose_keypoint_coords = new float[max_pose_detections][num_keypoints][2];
        float[][] pose_twod_array;
        float[][][] pose_key_array;
        int pose_key_ind = 0;

        multi_pose_scores pose_score_array;
        ArrayList<scored_parts_data>    scored_parts;
        targets_array   keypoint_targets;

        print_debug(TAG, "inside multiposes");

        scored_parts = build_part_with_score_fast(score_threshold, heatmapArr);

        Collections.sort(scored_parts, new Comparator<scored_parts_data>() {
            @Override
            public int compare(scored_parts_data s1, scored_parts_data s2) {
                int val = Float.compare(s1.scores, s2.scores);
                if (val == 0){
                    return 0;
                }
                else if(val == 1){
                    return -1;
                }
                else {
                    return 1;
                }
            }
        });

        for (int i=0; i< scored_parts.size(); i++){
            print_debug(TAG, "new8: Scored parts desc:" + i + ":" + scored_parts.get(i).scores + ":" + scored_parts.get(i).keypoint_id + ":" + scored_parts.get(i).arr_x
                    + ":" + scored_parts.get(i).arr_y );
        }
        print_debug(TAG, "new8: New Scored parts:" + scored_parts.size());

        height = heatmapArr.length;
        width = heatmapArr[height-1].length;
        num_keypoints = heatmapArr[height-1][width-1].length;

        int offset_length = offsetArr[height-1][width-1].length;
        int offset_half = offset_length/2;

        int disp_fwd_length = dispFwdArr[height-1][width-1].length;
        int disp_fwd_half = disp_fwd_length/2;

        int disp_bwd_length = dispBwdArr[height-1][width-1].length;
        int disp_bwd_half = disp_bwd_length/2;

        for(int i = 0; i<height; i++)
        {  //modify saras check if any function for swapping axes
            for(int j = 0; j<width; j++)
            {
                count_o = 0;
                for(int k = 0; k < offset_length; k++)
                {
                    k_offset = k - (count_o * offset_half);
                    offsetNewArr[i][j][k_offset][count_o] = offsetArr[i][j][k];
                    //print_debug(TAG, "new8: Offset new values:" + offsetNewArr[i][j][k_offset][count_o] + "," + offsetArr[i][j][k]);
                    if (k == (offset_half-1))
                    {
                        count_o++;
                    }
                }
            }
        }

        for(int i = 0; i<height; i++)
        {  //modify saras check if any function for swapping axes
            for(int j = 0; j<width; j++)
            {
                count_f = 0;
                for(int k = 0; k < disp_fwd_length; k++)
                {
                    k_disp_f = k - (count_f * disp_fwd_half);
                    dispNewFwdArr[i][j][k_disp_f][count_f] = dispFwdArr[i][j][k];
                    if (k == (disp_fwd_half-1))
                    {
                        count_f++;
                    }
                }
            }
        }

        for(int i = 0; i<height; i++)
        {  //modify saras check if any function for swapping axes
            for(int j = 0; j<width; j++)
            {
                count_b = 0;
                for(int k = 0; k < disp_bwd_length; k++)
                {
                    k_disp_b = k - (count_b * disp_bwd_half);
                    dispNewBwdArr[i][j][k_disp_b][count_b] = dispBwdArr[i][j][k];
                    if (k == (disp_bwd_half-1))
                    {
                        count_b++;
                    }
                }
            }
        }

        print_debug(TAG,"new8: height,width,length" + height + "," + width + "," + offset_length);
        print_debug(TAG, "new8: Offset new array length:" + offsetNewArr.length + "," + offsetNewArr[0].length + ","+ offsetNewArr[0][0].length + ","+ offsetNewArr[0][0][0].length);
        print_debug(TAG, "new8: dispFwdArr new array length:" + dispNewFwdArr.length + "," + dispNewFwdArr[0].length + ","+ dispNewFwdArr[0][0].length + ","+ dispNewFwdArr[0][0][0].length);
        print_debug(TAG, "new8: dispBwdArr new array length:" + dispNewBwdArr.length + "," + dispNewBwdArr[0].length + ","+ dispNewBwdArr[0][0].length + ","+ dispNewBwdArr[0][0][0].length);
        print_debug(TAG, "new8: Offset new array values:" + offsetNewArr[0][0][0][0] +":" + offsetNewArr[0][0][0][1]);
        print_debug(TAG, "new8: dispFwdArr new array values:" + dispNewFwdArr[0][0][0][0] +":" + dispNewFwdArr[0][0][0][1]);
        print_debug(TAG, "new8: dispBwdArr new array values:" + dispNewBwdArr[0][0][0][0] +":" + dispNewBwdArr[0][0][0][1]);

        root_image_coords = new float[2];

        for (int parts_ind = 0; parts_ind < scored_parts.size(); parts_ind++)
        {
            root_score = scored_parts.get(parts_ind).scores;
            root_id = scored_parts.get(parts_ind).keypoint_id;
            print_debug(TAG, "new8: root_score:" + root_score + ":root_id:" + root_id);

            root_coord_x = scored_parts.get(parts_ind).arr_x;
            root_coord_y = scored_parts.get(parts_ind).arr_y;

            print_debug(TAG, "new8: root_coord_x:" + root_coord_x + ":root_coord_y:" + root_coord_y);
            root_image_coords[0] = (root_coord_x * output_stride) + offsetNewArr[root_coord_x][root_coord_y][root_id][0];
            root_image_coords[1] = (root_coord_y * output_stride) + offsetNewArr[root_coord_x][root_coord_y][root_id][1];
            print_debug(TAG, "new8: root_image_coords:" + root_image_coords[0] + "," + root_image_coords[1]);
            print_debug(TAG, "new8: pose_count first:" + pose_count);

            pose_twod_array = new float[pose_count][2];

            print_debug(TAG, "new8: pose_twod_array in else here");
            for (int i = 0; i < pose_count; i++)
            {
                print_debug(TAG, "new6: not pose_count1:" + pose_count);
                pose_twod_array[i] = pose_keypoint_coords[i][root_id];
                print_debug(TAG, "new8: pose_twod_array 0" + ":" + pose_keypoint_coords[i][root_id][0] + ":" + pose_keypoint_coords[i][root_id][1]);
            }

            print_debug(TAG, "new8: pose_twod_array" + ":" + pose_twod_array.length);
            print_debug(TAG, "new8: pose_twod_array 1" + ":" + root_image_coords[0] + ":" + root_image_coords[1]);


            if (within_nms_radius_fast(pose_twod_array, root_image_coords))
            {
                print_debug(TAG, "new8: Value in if: root_id:" + root_id + ":" + "True");
                continue;
            }

            print_debug(TAG, "new8: Value outside if: root_id:" + root_id + ":" + "False");

            keypoint_targets = decode_pose(root_score, root_id, root_image_coords[0], root_image_coords[1], heatmapArr, offsetNewArr, output_stride, dispNewFwdArr, dispNewBwdArr);

            for (int i=0; i< TemplateClass.num_keypoints; i++){
                print_debug(TAG, "new8: keypoint_targets: for root_id:" + root_id + "is:" + keypoint_targets.scores[i]);
            }

            pose_key_array = new float[pose_count][num_keypoints][2];

            for (int i = 0; i < pose_count; i++) {
                pose_key_array[i] = pose_keypoint_coords[i];
                print_debug(TAG, "new6: not pose_count2:" + pose_count);
            }

            print_debug(TAG, "new8: pose_key_array:" + pose_key_array.length);
            print_debug(TAG, "new6: not pose_count3:" + pose_count);

            pose_score = get_instance_score_fast(pose_key_array, keypoint_targets.scores, keypoint_targets.image_coord);

            print_debug(TAG, "new8: min_pose_score:" + min_pose_score);
            print_debug(TAG, "new8: pose_score:" + pose_score);

            if ((min_pose_score == 0.0f) | (pose_score >= min_pose_score))
            {

                pose_scores[pose_count] = pose_score;
                pose_keypoint_scores[pose_count] = keypoint_targets.scores;
                pose_keypoint_coords[pose_count] = keypoint_targets.image_coord;
                pose_count++;
                pose_key_ind++;
                print_debug(TAG, "new8: pose_score in loop:" + pose_score + ":for root_id:" + root_id + ":root_score:" + root_score);
                print_debug(TAG, "new8: pose_count in loop:" + pose_count);
            }

            if (pose_count >= max_pose_detections) {
                print_debug(TAG, "new8: reached max count:" + pose_count + ":" + pose_scores.length);
                break;
            }
        }

        for (int i=0; i<pose_count; i++)
        {
            if (pose_count == 0)
            {
                break;
            }
            else
            {
                TemplateClass.poses_detected = true;

                Log.d(TAG, "new9: Pose Scores:" + pose_scores[i]);
                for(int k=0; k < pose_keypoint_scores[i].length; k++)
                {
                    Log.d(TAG, "new9: Keypoint : " + part_names[k] + ", score : " + pose_keypoint_scores[i][k] + ", coords : " + pose_keypoint_coords[i][k][0]
                            + ", " + pose_keypoint_coords[i][k][1]);
                }

            }
        }

        pose_score_array = new TemplateClass.multi_pose_scores(pose_scores, pose_keypoint_scores, pose_keypoint_coords);
        return pose_score_array;
    }
}

