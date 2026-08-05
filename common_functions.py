import numpy as np
from collections import deque
import smtplib
from email.mime.text import MIMEText
from datetime import datetime


def landmark_names(pose_landmarks):
    nose = pose_landmarks.landmark[0]
    right_shoulder = pose_landmarks.landmark[12]
    left_shoulder = pose_landmarks.landmark[11]
    right_elbow = pose_landmarks.landmark[14]
    left_elbow = pose_landmarks.landmark[13]
    right_wrist = pose_landmarks.landmark[16]
    left_wrist = pose_landmarks.landmark[15]
    right_hip = pose_landmarks.landmark[24]
    left_hip = pose_landmarks.landmark[23]
    right_knee = pose_landmarks.landmark[26]
    left_knee = pose_landmarks.landmark[25]
    left_heel = pose_landmarks.landmark[29]
    right_heel = pose_landmarks.landmark[12]

    return nose, right_shoulder, left_shoulder, right_elbow, left_elbow, right_wrist, left_wrist, right_hip, left_hip, right_knee, left_knee, right_heel, left_heel


def center_hip(right_hip, left_hip):
    center_hip_x = (right_hip.x + left_hip.x) / 2
    center_hip_y = (right_hip.y + left_hip.y) / 2

    return center_hip_x, center_hip_y


def center_chest(right_shoulder, left_shoulder):
    center_chest_x = (right_shoulder.x + left_shoulder.x) / 2
    center_chest_y = (right_shoulder.y + left_shoulder.y) / 2

    return center_chest_x, center_chest_y


def center_feet(right_heel, left_heel):
    center_feet_x = (right_heel.x + left_heel.x) / 2
    center_feet_y = (right_heel.y + left_heel.y) / 2

    return center_feet_x, center_feet_y


def landmark_sets(center_chest_x, center_chest_y, center_hip_x, center_hip_y, nose, right_elbow, left_elbow, right_wrist, left_wrist, right_knee, left_knee, right_heel, left_heel):
    landmark_set_x = (nose.x, center_chest_x, right_elbow.x, left_elbow.x, right_wrist.x, left_wrist.x, center_hip_x, right_knee.x, left_knee.x, right_heel.x, left_heel.x)
    landmark_set_y = (nose.y, center_chest_y, right_elbow.y, left_elbow.y, right_wrist.y, left_wrist.y, center_hip_y, right_knee.y, left_knee.y, right_heel.y, left_heel.y)

    #normalizing landmarks around the center of the hip
    normalized_landmark_array_x = np.asarray(landmark_set_x) - center_hip_x
    normalized_landmark_array_y = np.asarray(landmark_set_y) - center_hip_y

    return normalized_landmark_array_x, normalized_landmark_array_y


def body_aspect_ratio(nose, center_feet_y, right_shoulder, left_shoulder):
    body_height = nose.y - center_feet_y
    body_width = left_shoulder.x - right_shoulder.x
    body_aspect_ratio = body_height / body_width

    return body_aspect_ratio


def torso_angle_to_ground(center_chest_x, center_chest_y, center_hip_x, center_hip_y):
    #Returns torso angle relative to the ground (horizontal).
    #0 degrees  -> body horizontal (lying)
    #90 degrees -> body vertical (standing)

    # torso vector components (hip -> chest)
    torso_x = center_chest_x - center_hip_x
    torso_y = center_chest_y - center_hip_y

    # angle relative to horizontal axis
    torso_angle_rad = np.arctan2(abs(torso_y), abs(torso_x))
    torso_angle_deg = np.degrees(torso_angle_rad)

    return torso_angle_deg


def collect_data_over_frames(data_1, normalized_landmark_array_x, normalized_landmark_array_y, torso_angle_deg, body_aspect_ratio):
    frame_features = (normalized_landmark_array_x.tolist() + normalized_landmark_array_y.tolist() + [torso_angle_deg, body_aspect_ratio])
    
    data_1.extend(frame_features)

    return data_1


def send_fall_email():
    #Enter the emails you want to use to send the fall notification from and to
    sender_email = "enter sender email"
    receiver_email = "enter receiver email"

    #Get a gmail app password for your sender email
    app_password = "email app password"

    subject = "Fall Detection Alert"
    body = "A fall was detected"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("Email sent successfully")

    except Exception as e:
        print("Email failed:", e)


    

    

