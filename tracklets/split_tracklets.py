from .splitters.jersey_splitter import JerseySplitter
from collections import Counter
import numpy as np


def split_tracklets(tracklets, splitter_cfg):
    """ Split tracklets where jersey number changes and persists """
    jersey_splitter = JerseySplitter(
        min_fragment_length=splitter_cfg.jersey_min_fragment,
        min_persistence=splitter_cfg.jersey_min_persistence,
        lookahead=splitter_cfg.jersey_lookahead,
    )
    
    initial_count = len(tracklets)
    initial_detections = sum(len(t.frames) for t in tracklets.values())
    jersey_splits = 0
    
    max_existing_id = max(tracklets.keys())
    next_available_id = max_existing_id + 1
    
    new_tracklets = {}
    
    for track_id, tracklet in tracklets.items():
        # Check if this tracklet has multiple GT jerseys (potential ID switch)
        gt_jerseys = tracklet.gt_attributes.get('jerseys', [])
        unique_gt = set(j for j in gt_jerseys if j != 'unknown')
                
        fragments = jersey_splitter.split_tracklet(tracklet, next_available_id)
        
        if fragments:
            for fragment in fragments:
                new_tracklets[fragment.track_id] = fragment
                next_available_id = max(next_available_id, fragment.track_id + 1)
        
        else:
            new_tracklets[tracklet.track_id] = tracklet
    
    # Compute final jersey for all tracklets
    for tracklet in new_tracklets.values():
        tracklet.final_jersey = jersey_splitter.get_final_value(tracklet)
    

    
    return new_tracklets