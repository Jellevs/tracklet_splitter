from .splitters.jersey_splitter import JerseySplitter


def split_tracklets(tracklets, splitter_cfg):
    """ Split tracklets where jersey number changes and persists """
    jersey_splitter = JerseySplitter(splitter_cfg)

    max_existing_id = max(tracklets.keys())
    next_available_id = max_existing_id + 1
    
    new_tracklets = {}
    
    for track_id, tracklet in tracklets.items():       
        fragments = jersey_splitter.split_tracklet(tracklet, next_available_id)
        
        if fragments:
            for fragment in fragments:
                print(f"splitted tracklet {track_id} into {len(fragments)} fragments: {fragment.to_dict().get('pred_attributes')['jerseys']}")
                new_tracklets[fragment.track_id] = fragment
                next_available_id = max(next_available_id, fragment.track_id + 1)
        
        else:
            new_tracklets[tracklet.track_id] = tracklet
    
    
    return new_tracklets