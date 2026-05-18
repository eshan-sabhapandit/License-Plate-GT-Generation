import cv2
import numpy as np
import time
import os
import json

BOS_CAM1_ZONE = (100, 450, 500, 270)
BOS_CAM2_ZONE = (100, 400, 500, 320)



def detect_motion_timestamps(video_path, zone=(100, 400, 500, 320), threshold=10000, history=1000, varThreshold=50):
    """
    Detects when vehicle enters a specified zone in image and returns list of timestamps. Logic is to use background 
    subtraction to remove the background from the image and then check if the zone is motion.
    
    Args:
        video_path (str): Path to the MP4 file.
        zone (tuple): (x, y, w, h) of the trigger zone in image. Note: (x, y) is the top-left corner of the zone.
        threshold (int): Minimum pixel change to trigger detection.
        history (int): Number of frames to consider for background subtraction. Used by MOG2 algorithm. Default is 500.
        varThreshold (int): Threshold for background subtraction. Used by MOG2 algorithm. Default is 50.

    Returns:
        timestamps_mm_ss_entry: Entry times in MM:SS (first spurious event dropped).
        timestamps_seconds_entry: Entry times in seconds.
        timestamps_mm_ss_exit: Exit times in MM:SS (first spurious event dropped).
        timestamps_seconds_exit: Exit times in seconds.
        exit_seconds_by_detection: Mapping ``detection_index -> exit_seconds`` for each
            kept exit (0-based index matches position in the exit lists above).
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a background subtractor to remove the background from the image using MOG2 algorithm
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=True)
    
    timestamps_mm_ss_entry = []   # List of timestamps when vehicle enters the zone in MM:SS format
    timestamps_seconds_entry = [] # List of seconds when vehicle enters the zone
    timestamps_mm_ss_exit = []   # List of timestamps when vehicle exits the zone in MM:SS format
    timestamps_seconds_exit = [] # List of seconds when vehicle exits the zone
    is_detecting = False # Prevents multiple detections for the same vehicle
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Current time in second
        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_seconds = current_ms / 1000.0

        # Background Subtraction on the whole frame, returns a binary mask where background is black
        fgmask = fgbg.apply(frame)
        
        # Clean noise using morphological opening
        kernel = np.ones((5,5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # Extract the zone from the binary mask to detect if vehicle (foreground) is in the zone
        x, y, w, h = zone
        mask_zone = fgmask[y:y+h, x:x+w]
        motion_score = np.sum(mask_zone > 0)    # Number of foreground pixels in the zone

        # Detection Logic: If the motion score is greater than the threshold, then the vehicle is in the zone (foreground)
        if motion_score > threshold:
            if not is_detecting:
                # Convert seconds to MM:SS format.
                minutes = int(current_seconds // 60)
                seconds = int(current_seconds % 60)
                timestamps_mm_ss_entry.append(f"{minutes:02d}:{seconds:02d}")
                timestamps_seconds_entry.append(current_seconds)
                is_detecting = True

        else:
            # Reset once the vehicle leaves the zone (low motion score), add the exit timestamp to the list
            if motion_score < (threshold * 10) and is_detecting:
                minutes = int(current_seconds // 60)
                seconds = int(current_seconds % 60)
                timestamps_mm_ss_exit.append(f"{minutes:02d}:{seconds:02d}")
                timestamps_seconds_exit.append(current_seconds)
                is_detecting = False

    cap.release()

    # Remove the first timestamp as it is just 00:00
    timestamps_mm_ss_entry = timestamps_mm_ss_entry[1:]
    timestamps_seconds_entry = timestamps_seconds_entry[1:]
    timestamps_mm_ss_exit = timestamps_mm_ss_exit[1:]
    timestamps_seconds_exit = timestamps_seconds_exit[1:]

    # Convert lists to a dictionary where the key is the MM:SS timestamp and the value is the seconds timestamp
    timestamps = dict(zip(timestamps_mm_ss_exit, timestamps_seconds_exit))
    
    return timestamps


if __name__ == "__main__":

    video_1 = "videos/192-168-100-22/11/"
    video_2 = "videos/192-168-100-32/11/"

    gate_zone_1 = BOS_CAM1_ZONE
    gate_zone_2 = BOS_CAM2_ZONE

    # Step 1: Create a json file to store the exit timestamps
    exit_timestamps_11_1 = {}
    exit_timestamps_11_2 = {}

    # Step 2: Run the vehicle detection function on all videos in the directory
    tic = time.time()
    for video in os.listdir(video_1):
        video_path = os.path.join(video_1, video)
        # Check if video is a mp4 file
        if video.endswith('.mp4'):
            exit_timestamps_11_1[video] = detect_motion_timestamps(video_path, gate_zone_1)
        else:
            continue

    for video in os.listdir(video_2):
        video_path = os.path.join(video_2, video)
        # Check if video is a mp4 file
        if video.endswith('.mp4'):
            exit_timestamps_11_2[video] = detect_motion_timestamps(video_path)
        else:
            continue

    # Step 3: Save the exit timestamps to the json file
    with open('exit_timestamps_11_1.json', 'w') as f:
        json.dump(exit_timestamps_11_1, f)
    with open('exit_timestamps_11_2.json', 'w') as f:
        json.dump(exit_timestamps_11_2, f)
    
    toc = time.time()
    print(f"Time taken: {toc - tic} seconds")