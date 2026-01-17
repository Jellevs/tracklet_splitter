# import cv2
# import supervision as sv
# import numpy as np 
# from tqdm import tqdm


# def visualize_tracklets(images, tracklets_dict, output_path, title="tracklets", show_teams=True):
#     """Visualize tracklets"""

#     frame_to_detections = tracklets_to_frame_detections(tracklets_dict)
    
#     # Create video
#     first_frame = cv2.imread(str(images[0]))
#     height, width = first_frame.shape[:2]
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = 25
#     out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
#     bbox_annotator = sv.BoxAnnotator(
#         color_lookup=sv.ColorLookup.TRACK,
#         thickness=1
#     )
    
#     label_annotator = sv.LabelAnnotator(
#         color_lookup=sv.ColorLookup.TRACK,
#         text_scale=0.25,
#         text_thickness=1,
#         text_padding=1,
#         border_radius=1
#     )

    
#     for frame_idx, image_path in tqdm(enumerate(images), total=len(images), desc=f"Visualizing {title}"):
#         image = cv2.imread(str(image_path))
        
#         detections_list = frame_to_detections.get(frame_idx, [])
        
#         if not detections_list:
#             out.write(image)
#             continue
        

#         xyxy = np.array([d['bbox'] for d in detections_list])
#         confidence = np.array([d['score'] for d in detections_list])
#         tracker_id = np.array([d['tracker_id'] for d in detections_list])

#         detections = sv.Detections(
#             xyxy=xyxy,
#             confidence=confidence,
#             tracker_id=tracker_id
#         )
        
#         labels = []
#         for detection in detections_list:


#             # Ground Truth attributes
#             tracker_id = detection['tracker_id']
#             parent_id = detection.get('parent_id')

#             gt_jersey = detection.get('gt_jersey')
#             gt_jersey = str(gt_jersey) if gt_jersey else "X"
            
#             # Team
#             gt_team = detection.get('gt_team')
#             gt_team = str(gt_team) if gt_team is not None else "X"

#             # Role
#             gt_role = detection.get('gt_role', -1)
#             gt_role = str(gt_role) if gt_role is not None else "X"


#             # Predicted attributes
#             # pred_jersey_window = detection.get('final_jersey')
#             # pred_jersey_window = str(pred_jersey_window) if pred_jersey_window is not None else "X"
            
#             pred_jersey_frame = detection.get('pred_jersey')
#             pred_jersey_frame = str(pred_jersey_frame) if pred_jersey_frame is not None else "X"

#             pred_team_frame = detection.get('pred_team')
#             pred_team_frame = str(pred_team_frame) if pred_team_frame is not None else "X"

#             # label = f"{tracker_id}|{gt_jersey}|{gt_team}|{gt_role}:pred{pred_jersey_frame}|{pred_team_frame}"
#             label = f"{pred_jersey_frame}|{pred_team_frame}"
                      
#             labels.append(label)

#         annotated = bbox_annotator.annotate(image.copy(), detections)
#         annotated = label_annotator.annotate(annotated, detections, labels=labels)
        
#         out.write(annotated)
    
#     out.release()
#     print(f"Video saved to: {output_path}")



# def tracklets_to_frame_detections(tracklets_dict):
#     """Convert tracklets to frame-by-frame detections"""
#     frame_to_detections = {}
    
#     for tracker_id, tracklet in tracklets_dict.items():
#         for i in range(len(tracklet.frames)):
#             frame_idx = tracklet.frames[i]

#             if frame_idx not in frame_to_detections:
#                 frame_to_detections[frame_idx] = []
            
#             detection_data = {
#                 'tracker_id': tracker_id,
#                 'bbox': tracklet.bboxes[i],
#                 'score': tracklet.scores[i],
#             }


#             # Ground Truth attributes
#             # GT jersey
#             gt_jerseys = tracklet.gt_attributes.get('jerseys')
#             if gt_jerseys is not None and len(gt_jerseys) > 0 and i < len(gt_jerseys):
#                 gt_jersey = gt_jerseys[i]
#                 if gt_jersey and gt_jersey != 'unknown':
#                     detection_data['gt_jersey'] = str(gt_jersey)

#             # GT Team
#             gt_teams = tracklet.gt_attributes.get('teams')
#             if gt_teams is not None and len(gt_teams) > 0 and i < len(gt_teams):
#                 gt_team = gt_teams[i]
#                 if gt_team and gt_team != 'unknown':
#                     detection_data['gt_team'] = str(gt_team)
            
#             # GT Role
#             gt_roles = tracklet.gt_attributes.get('roles')
#             if gt_roles is not None and len(gt_roles) > 0 and i < len(gt_roles):
#                 gt_role = gt_roles[i]
#                 if gt_role and gt_role != 'unknown':
#                     detection_data['gt_role'] = str(gt_role)


#             # Predicted attributes
#             # Jersey: use final jersey, otherwise per frame
#             if tracklet.final_jersey is not None:
#                 detection_data['final_jersey'] = tracklet.final_jersey
            
#             jerseys = tracklet.pred_attributes.get('jerseys', [])
#             if jerseys is not None and len(jerseys) > 0 and i < len(jerseys):
#                 jersey = jerseys[i]
#                 if not (isinstance(jersey, float) and np.isnan(jersey)):
#                     detection_data['pred_jersey'] = int(jersey)


#             # Team: use final team, otherwise per frame
#             if tracklet.final_team is not None:
#                 detection_data['final_team'] = tracklet.final_team
            
#             teams = tracklet.pred_attributes.get('teams', [])
#             if teams is not None and len(teams) > 0 and i < len(teams):
#                 team = teams[i]
#                 if not np.isnan(team):
#                     detection_data['pred_team'] = int(team)

            
#             frame_to_detections[frame_idx].append(detection_data)

#     return frame_to_detections



"""
Visualization utilities for tracklet analysis.
"""
import cv2
import supervision as sv
import numpy as np 
from tqdm import tqdm
from collections import Counter


def compute_window_consensus(values, idx, window_size, threshold=0.5):
    """
    Compute window consensus at a specific index.
    
    Args:
        values: List of values (can contain NaN/None)
        idx: Index to compute consensus around
        window_size: Size of sliding window
        threshold: Minimum ratio for consensus
    
    Returns:
        Consensus value or None if no consensus
    """
    n = len(values)
    half_window = window_size // 2
    
    start = max(0, idx - half_window)
    end = min(n, idx + half_window + 1)
    
    window = values[start:end]
    
    # Filter valid values
    valid = []
    for val in window:
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            valid.append(val)
    
    if not valid:
        return None
    
    counter = Counter(valid)
    most_common, count = counter.most_common(1)[0]
    
    if count / len(valid) >= threshold:
        return most_common
    
    return None


def visualize_tracklets(images, tracklets_dict, output_path, title="tracklets"):
    """ Visualize tracklets with predictions """

    frame_to_detections = tracklets_to_frame_detections(
        tracklets_dict, 
    )
    
    # Create video
    first_frame = cv2.imread(str(images[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    bbox_annotator = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
        thickness=1
    )
    
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.TRACK,
        text_scale=0.25,
        text_thickness=1,
        text_padding=1,
        border_radius=1
    )

    for frame_idx, image_path in tqdm(enumerate(images), total=len(images), desc=f"Visualizing {title}"):
        image = cv2.imread(str(image_path))
        
        detections_list = frame_to_detections.get(frame_idx, [])
        
        if not detections_list:
            out.write(image)
            continue

        xyxy = np.array([d['bbox'] for d in detections_list])
        confidence = np.array([d['score'] for d in detections_list])
        tracker_id = np.array([d['tracker_id'] for d in detections_list])

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            tracker_id=tracker_id
        )
        
        labels = []
        for detection in detections_list:

            tracker_id = detection['tracker_id']
            parent_id = detection.get('parent_id')
            
            # Ground Truth
            gt_jersey = detection.get('gt_jersey', 'X')
            gt_team = detection.get('gt_team', 'X')
            gt_role = detection.get('gt_role', 'X')
            
            # Per-frame predictions
            pred_jersey_frame = detection.get('pred_jersey')
            pred_jersey_frame = str(pred_jersey_frame) if pred_jersey_frame is not None else "X"
            
            pred_team_frame = detection.get('pred_team')
            pred_team_frame = str(pred_team_frame) if pred_team_frame is not None else "X"
            
            # Window predictions
            pred_jersey_window = detection.get('pred_jersey_window')
            pred_jersey_window = str(int(pred_jersey_window)) if pred_jersey_window is not None else "X"
            
            pred_team_window = detection.get('pred_team_window')
            pred_team_window = str(int(pred_team_window)) if pred_team_window is not None else "X"
            
            
            label = f"{tracker_id}|{pred_jersey_frame}|{pred_team_frame}"
                                   
            labels.append(label)

        annotated = bbox_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)
        
        out.write(annotated)
    
    out.release()
    print(f"Video saved to: {output_path}")


def tracklets_to_frame_detections(tracklets_dict):
    """ Convert tracklets to frame-by-frame detections """
    frame_to_detections = {}

    for tracker_id, tracklet in tracklets_dict.items():
        for i in range(len(tracklet.frames)):
            frame_idx = tracklet.frames[i]

            if frame_idx not in frame_to_detections:
                frame_to_detections[frame_idx] = []
            
            detection_data = {
                'tracker_id': tracker_id,
                'bbox': tracklet.bboxes[i],
                'score': tracklet.scores[i],
            }
            

            # Ground Truth attributes
            gt_jerseys = tracklet.gt_attributes.get('jerseys')
            if gt_jerseys is not None and len(gt_jerseys) > 0 and i < len(gt_jerseys):
                gt_jersey = gt_jerseys[i]
                if gt_jersey and gt_jersey != 'unknown':
                    detection_data['gt_jersey'] = str(gt_jersey)

            gt_teams = tracklet.gt_attributes.get('teams')
            if gt_teams is not None and len(gt_teams) > 0 and i < len(gt_teams):
                gt_team = gt_teams[i]
                if gt_team and gt_team != 'unknown':
                    detection_data['gt_team'] = str(gt_team)
            
            gt_roles = tracklet.gt_attributes.get('roles')
            if gt_roles is not None and len(gt_roles) > 0 and i < len(gt_roles):
                gt_role = gt_roles[i]
                if gt_role and gt_role != 'unknown':
                    detection_data['gt_role'] = str(gt_role)


            # Per-frame predictions
            jerseys = tracklet.pred_attributes.get('jerseys', [])
            if jerseys is not None and len(jerseys) > 0 and i < len(jerseys):
                jersey = jerseys[i]
                if not (isinstance(jersey, float) and np.isnan(jersey)):
                    detection_data['pred_jersey'] = int(jersey)

            teams = tracklet.pred_attributes.get('teams', [])
            if teams is not None and len(teams) > 0 and i < len(teams):
                team = teams[i]
                if not (isinstance(team, float) and np.isnan(team)):
                    detection_data['pred_team'] = int(team)
            

            
            frame_to_detections[frame_idx].append(detection_data)

    return frame_to_detections