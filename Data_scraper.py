import os
import math
import mediapipe as mp
import cv2 as cv
import numpy as np
from collections import deque
import common_functions

model_ran = 0

dataset_X = []
dataset_y = []

data_1 = deque([], maxlen= 720)

video_folder = "video_folder"

#scans the directory, filters only video formats, builds full file paths
video_files = [
    os.path.join(video_folder, f)
    for f in os.listdir(video_folder)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]



for video_file in video_files:
    label = 1 if "fall" in video_file.lower() else 0

    vid = cv.VideoCapture(video_file)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    frame_number = -1


    while True:
        success, frame = vid.read()
        # video ended or camera failed
        if not success:
            break   

        #Keep track of how many frames have passed by numbering each frame
            
        frame_number += 1

        #opencv uses BGR but mediapipe uses RGB. Need to convert
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        pose_landmarks = results.pose_landmarks

        if not results.pose_landmarks:
            continue


        #Every other frame (to make the program more computationally efficient) rename the pose landmarks detected in the video
        #and create new ones needed to preform calculations (center_hip, center_feet, etc...)
        if frame_number % 2 == 0:
            (nose,
            right_shoulder, left_shoulder,
            right_elbow, left_elbow,
            right_wrist, left_wrist,
            right_hip, left_hip,
            right_knee, left_knee,
            right_heel, left_heel) = common_functions.landmark_names(results.pose_landmarks)
            
            center_hip_x, center_hip_y = common_functions.center_hip(right_hip, left_hip)

            center_chest_x, center_chest_y = common_functions.center_chest(right_shoulder, left_shoulder)

            center_feet_x, center_feet_y = common_functions.center_feet(right_heel, left_heel)


            #Create a set of landmarks normalized around the center_hip landmark
            (normalized_x,
            normalized_y) = common_functions.landmark_sets(
            center_chest_x, center_chest_y,
            center_hip_x, center_hip_y,
            nose,
            right_elbow, left_elbow,
            right_wrist, left_wrist,
            right_knee, left_knee,
            right_heel, left_heel)

            body_ratio= common_functions.body_aspect_ratio(nose, center_feet_y, right_shoulder, left_shoulder)

            torso_angle_deg = common_functions.torso_angle_to_ground(center_chest_x, center_chest_y, center_hip_x, center_hip_y)

            data_1 = common_functions.collect_data_over_frames(
            data_1,
            normalized_x,
            normalized_y,
            torso_angle_deg,
            body_ratio)



            #If the deque of collected data is full to the length of data we want to feed into the machine learning model, append that data to the dataset lists.
            if len(data_1) == 720:
                # convert deque → fixed numpy array
                sample = np.array(data_1, dtype=np.float32)

                # store dataset sample
                dataset_X.append(sample)
                dataset_y.append(label)

                print(f"sample collected from {video_file}")

                model_ran += 1
                data_1.clear()
                break


            if results.pose_landmarks:    
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            #opens window with image
            cv.imshow('Data Video', frame)




        if cv.waitKey(1) == ord('q') or model_ran == len(video_files):
            break
cv.destroyAllWindows()


#Once we collected the data from all of the video's, save the datasets to an .npz to be loaded in to the file where the model will be trained 
X = np.array(dataset_X, dtype=np.float32)
y = np.array(dataset_y, dtype=np.int64)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

np.savez("fall_dataset.npz", X=X, y=y)