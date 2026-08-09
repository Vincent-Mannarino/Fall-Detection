import math
import mediapipe as mp
import cv2 as cv
import numpy as np
from collections import deque
import common_functions
import torch
from torch import nn
import smtplib
from email.mime.text import MIMEText
import datetime
import time

frame_number = -1
target_frame = 0
window_stride = 6

fall_streak = 0
required_streak = 3

last_fall_time = 0
fall_cooldown = 5  # seconds

data_1 = deque([], maxlen= 720)


#Copy the chosen model class and load the model parameters 
class FallDetectionModel(nn.Module):
  def __init__(self, input_features = 720, hidden_units = 16, output_features = 1):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
      nn.Linear(in_features=input_features, out_features=hidden_units),

      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      
      nn.Linear(in_features=hidden_units, out_features=output_features)
  )
  def forward(self, x):
    return self.linear_layer_stack(x)

device = torch.device("cpu")
model = FallDetectionModel().to(device)
model.load_state_dict(torch.load("fall_model.pth", map_location=device))
model.eval()


#Chose the camera you want to use (0 for defult laptop camera, 1 for additional attached camera, etc...) 
vid = cv.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils


while True:
    success, frame = vid.read()

    if not success or frame is None:
        break


    #Keep track of how many frames have passed by numbering each frame
    frame_number += 1

    #opencv uses BGR but mediapipe uses RGB. Need to convert
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    pose_landmarks = results.pose_landmarks


    if pose_landmarks and frame_number % 2 == 0:
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


        if len(data_1) == 720 and frame_number >= target_frame:
            features = np.array(data_1, dtype=np.float32)
            input_tensor = torch.from_numpy(features)
            input_tensor = input_tensor.unsqueeze(0)  # add batch dimension
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                #Run the data through the model to get the fall confidence between 0 and 1
                confidence = torch.sigmoid(model(input_tensor))
                
            if confidence > 0.50:
                fall_streak += 1
            else:
                fall_streak = 0

            #The model will detect a fall multiple times in a single fall, and sometimes when no fall actually occurred.
            #To limit false detections, a fall is not reported unless the model detects enough falls in a row denoted by the 'required_streak' variable
            if fall_streak >= required_streak and time.time() - last_fall_time > fall_cooldown:
                fall_streak = 0

                print("fall detected")

                #Sends an email if a fall was detecteed
                common_functions.send_fall_email()
                last_fall_time = time.time()


            target_frame = frame_number + window_stride


        if results.pose_landmarks:    
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


    #opens window with image
    cv.imshow('Fall Detection', frame)

    if cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()