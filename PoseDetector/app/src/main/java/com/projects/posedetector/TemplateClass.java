package com.projects.posedetector;

import android.graphics.Bitmap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class TemplateClass {

    public static Bitmap bitmap;

    public static boolean poses_detected;

    public static class multi_pose_scores {
        public float[] scores;
        public float[][] pose_key_scores;
        public float[][][] pose_key_coords;

        public multi_pose_scores(float[] scores,
                                 float[][] pose_key_scores,
                                 float[][][] pose_key_coords)
        {
            this.scores = scores;
            this.pose_key_scores = pose_key_scores;
            this.pose_key_coords = pose_key_coords;
        }
    }

    public static class targets {
        public float scores;
        public float[] image_coord;

        public targets(float scores,
                       float[] image_coord){
            this.scores = scores;
            this.image_coord = image_coord;
        }
    }

    public static class targets_array {
        public float[] scores;
        public float[][] image_coord;

        public targets_array(float[] scores,
                             float[][] image_coord)
        {
            this.scores = scores;
            this.image_coord = image_coord;
        }
    }

    public static class scored_parts_data{
        public float scores;
        public int keypoint_id;
        public int arr_x;
        public int arr_y;

        public scored_parts_data(float scores,
                                 int keypoint_id,
                                 int arr_x,
                                 int arr_y)
        {
            this.scores = scores;
            this.keypoint_id = keypoint_id;
            this.arr_x = arr_x;
            this.arr_y = arr_y;
        }
    }

    public static class array_pos{
        public int x_pos;
        public int y_pos;

        public array_pos(int x_pos,
                         int y_pos)
        {
            this.x_pos = x_pos;
            this.y_pos = y_pos;
        }
    }

    public static multi_pose_scores multiPoseScores;

    public static float min_pose_score = 0.15f; //0.2495f;
    public static float min_part_score = 0.15f; //0.25f; //modify saras
    public static final float score_threshold = 0.5f; //0.516f;
    public static final int max_pose_detections = 10;
    public static final float nms_radius = 20;
    public static final int local_maximum_radius = 1;
    public static final int output_stride = 16; //16;
    public static final String model = "model-mobilenet_v1_050-" + output_stride + ".tflite"; //16;
    public static final float squared_nms_radius = nms_radius * nms_radius;
    public static int num_parts = 17;
    public static int num_edges = 16;
    public static final int INPUT_SIZE = 225; //113;
    public static final int ARR_RESOLUTION = 15; //((INPUT_SIZE-1)/output_stride)+1;

    public static HashMap<String, Integer> parts_ids = new HashMap<String, Integer>();
    public static List<String[]> connected_part_names = new ArrayList<>();
    public static List<Integer[]> connected_part_indices = new ArrayList<>();
    public static List<String[]> pose_chain = new ArrayList<>();
    public static List<Integer[]> parent_child_tuples = new ArrayList<>();

    public static String[] part_names = new String[] {"nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"};

    public static int num_keypoints = part_names.length;
    public static final int offset_res = num_keypoints * 2;

    public static String[] part_channels = new String[] {"left_face", "right_face", "right_upper_leg_front", "right_lower_leg_back", "right_upper_leg_back",
            "left_lower_leg_front", "left_upper_leg_front", "left_upper_leg_back", "left_lower_leg_back",
            "right_feet", "right_lower_leg_front", "left_feet", "torso_front", "torso_back", "right_upper_arm_front",
            "right_upper_arm_back", "right_lower_arm_back", "left_lower_arm_front", "left_upper_arm_front",
            "left_upper_arm_back", "left_lower_arm_back", "right_hand", "right_lower_arm_front", "left_hand"};


    public static void constant_var()
    {

        {
            parts_ids.put("nose",0);
            parts_ids.put("leftEye",1);
            parts_ids.put("rightEye",2);
            parts_ids.put("leftEar",3);
            parts_ids.put("rightEar",4);
            parts_ids.put("leftShoulder",5);
            parts_ids.put("rightShoulder",6);
            parts_ids.put("leftElbow",7);
            parts_ids.put("rightElbow",8);
            parts_ids.put("leftWrist",9);
            parts_ids.put("rightWrist",10);
            parts_ids.put("leftHip",11);
            parts_ids.put("rightHip",12);
            parts_ids.put("leftKnee",13);
            parts_ids.put("rightKnee",14);
            parts_ids.put("leftAnkle",15);
            parts_ids.put("rightAnkle",16);
        }

        String[] arr1 = {"leftHip", "leftShoulder"};
        String[] arr2 = {"leftElbow", "leftShoulder"};
        String[] arr3 = {"leftElbow", "leftWrist"};
        String[] arr4 = {"leftHip", "leftKnee"};
        String[] arr5 = {"leftKnee", "leftAnkle"};
        String[] arr6 = {"rightHip", "rightShoulder"};
        String[] arr7 = {"rightElbow", "rightShoulder"};
        String[] arr8 = {"rightElbow", "rightWrist"};
        String[] arr9 = {"rightHip", "rightKnee"};
        String[] arr10 = {"rightKnee", "rightAnkle"};
        String[] arr11 = {"leftShoulder", "rightShoulder"};
        String[] arr12 = {"leftHip", "rightHip"};

        connected_part_names.add(arr1);
        connected_part_names.add(arr2);
        connected_part_names.add(arr3);
        connected_part_names.add(arr4);
        connected_part_names.add(arr5);
        connected_part_names.add(arr6);
        connected_part_names.add(arr7);
        connected_part_names.add(arr8);
        connected_part_names.add(arr9);
        connected_part_names.add(arr10);
        connected_part_names.add(arr11);
        connected_part_names.add(arr12);

        Integer[] num1 = {11, 5};
        Integer[] num2 = {7, 5};
        Integer[] num3 = {7, 9};
        Integer[] num4 = {11, 13};
        Integer[] num5 = {13, 15};
        Integer[] num6 = {12, 6};
        Integer[] num7 = {8, 6};
        Integer[] num8 = {8, 10};
        Integer[] num9 = {12, 14};
        Integer[] num10 = {14, 16};
        Integer[] num11 = {5, 6};
        Integer[] num12 = {11, 12};

        connected_part_indices.add(num1);
        connected_part_indices.add(num2);
        connected_part_indices.add(num3);
        connected_part_indices.add(num4);
        connected_part_indices.add(num5);
        connected_part_indices.add(num6);
        connected_part_indices.add(num7);
        connected_part_indices.add(num8);
        connected_part_indices.add(num9);
        connected_part_indices.add(num10);
        connected_part_indices.add(num11);
        connected_part_indices.add(num12);


        String[] pose1 = {"nose", "leftEye"};
        String[] pose2 = {"leftEye", "leftEar"};
        String[] pose3 = {"nose", "rightEye"};
        String[] pose4 = {"rightEye", "rightEar"};
        String[] pose5 = {"nose", "leftShoulder"};
        String[] pose6 = {"leftShoulder", "leftElbow"};
        String[] pose7 = {"leftElbow", "leftWrist"};
        String[] pose8 = {"leftShoulder", "leftHip"};
        String[] pose9 = {"leftHip", "leftKnee"};
        String[] pose10 = {"leftKnee", "leftAnkle"};
        String[] pose11 = {"nose", "rightShoulder"};
        String[] pose12 = {"rightShoulder", "rightElbow"};
        String[] pose13 = {"rightElbow", "rightWrist"};
        String[] pose14 = {"rightShoulder", "rightHip"};
        String[] pose15 = {"rightHip", "rightKnee"};
        String[] pose16 = {"rightKnee", "rightAnkle"};


        pose_chain.add(pose1);
        pose_chain.add(pose2);
        pose_chain.add(pose3);
        pose_chain.add(pose4);
        pose_chain.add(pose5);
        pose_chain.add(pose6);
        pose_chain.add(pose7);
        pose_chain.add(pose8);
        pose_chain.add(pose9);
        pose_chain.add(pose10);
        pose_chain.add(pose11);
        pose_chain.add(pose12);
        pose_chain.add(pose13);
        pose_chain.add(pose14);
        pose_chain.add(pose15);
        pose_chain.add(pose16);


        Integer[] parent1 = {0, 1};
        Integer[] parent2 = {1, 3};
        Integer[] parent3 = {0, 2};
        Integer[] parent4 = {2, 4};
        Integer[] parent5 = {0, 5};
        Integer[] parent6 = {5, 7};
        Integer[] parent7 = {7, 9};
        Integer[] parent8 = {5, 11};
        Integer[] parent9 = {11, 13};
        Integer[] parent10 = {13, 15};
        Integer[] parent11 = {0, 6};
        Integer[] parent12 = {6, 8};
        Integer[] parent13 = {8, 10};
        Integer[] parent14 = {6, 12};
        Integer[] parent15 = {12, 14};
        Integer[] parent16 = {14, 16};

        parent_child_tuples.add(parent1);
        parent_child_tuples.add(parent2);
        parent_child_tuples.add(parent3);
        parent_child_tuples.add(parent4);
        parent_child_tuples.add(parent5);
        parent_child_tuples.add(parent6);
        parent_child_tuples.add(parent7);
        parent_child_tuples.add(parent8);
        parent_child_tuples.add(parent9);
        parent_child_tuples.add(parent10);
        parent_child_tuples.add(parent11);
        parent_child_tuples.add(parent12);
        parent_child_tuples.add(parent13);
        parent_child_tuples.add(parent14);
        parent_child_tuples.add(parent15);
        parent_child_tuples.add(parent16);

    }

}
