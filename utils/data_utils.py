import json
from pathlib import Path
from dataclasses import dataclass
import supervision as sv
import numpy as np
from tqdm import tqdm
import cv2

from detect_and_track.trackers.Deep_EIoU import tracker
from tracklets.tracklet import Tracklet


def load_images(img_dir):
    """ Load images """
    img_path = Path(img_dir)
    img_paths = sorted(img_path.glob("*.jpg"))
    return img_paths


def organize_detections_by_track(track_data):
    """" Convert detections into a dictionary of {track_id: tracklet} """
    tracklets_dict = {}
    
    for frame_data in track_data:
        frame_idx = frame_data['frame_idx']
        detections = frame_data['tracked_detections']

        for i in range(len(detections)):
            pred_track_id = detections.tracker_id[i]

            if pred_track_id not in tracklets_dict:
                tracklets_dict[pred_track_id] = Tracklet(track_id=pred_track_id)

            tracklets_dict[pred_track_id].append_from_detection(
                frame_idx=frame_idx,
                detection_idx=i,
                detections=detections
            )

    return tracklets_dict