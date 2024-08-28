from time import sleep

import cv2
import mediapipe as mp
import numpy as np
from numpy import ndarray
from typing import Optional
from one_euro_filter import OneEuroFilter
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Define the filter parameters
min_cutoff = 0.05
beta = 80.0
derivate_cutoff = 1.0

# Create an array to hold the filters
num_landmarks = 33
num_coordinates = 3  # x, y, z

# Use a nested list comprehension to create the 3D array of filters
filters = np.array([[
    OneEuroFilter(frequency=30, min_cutoff=min_cutoff, beta=beta, derivate_cutoff=derivate_cutoff,
                  to_print=(i == 0 and j == 0))
    for j in range(num_coordinates)]
    for i in range(num_landmarks)])

global_annotated_image: Optional[ndarray] = None


def get_object_scale(landmarks):
    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    object_width = x_max - x_min
    object_height = y_max - y_min

    return (object_width + object_height) / 2.0


def default_inference_draw(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global global_annotated_image
    pose_landmarks_list = result.pose_landmarks
    annotated_image = np.copy(output_image.numpy_view())

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        object_scale = get_object_scale(pose_landmarks)
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=filters[i][0].apply(landmark.x, timestamp_ms, object_scale),
                y=filters[i][1].apply(landmark.y, timestamp_ms, object_scale),
                z=filters[i][2].apply(landmark.z, timestamp_ms, object_scale))
            for i, landmark in enumerate(pose_landmarks)
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    global_annotated_image = np.copy(annotated_image)


options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=default_inference_draw
        )

with PoseLandmarker.create_from_options(options) as landmarker:
    frame = cv2.imread('Standing-Man.png')
    i = 0
    while True:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(image, i)
        i += 1
        if global_annotated_image is not None:
            cv2.imshow('Frame', global_annotated_image)

        # sleep(0.033)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

