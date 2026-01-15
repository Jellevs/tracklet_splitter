from tqdm import tqdm
import pickle
import cv2
import numpy as np

from .jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq
from .teamclassifier import TeamClassifier


def predict_attributes(images, tracklets, paths, jersey_cfg, device):
    cache_path = paths.set_cache_path("attributes", paths.sequence)

    # Check cache
    if cache_path and cache_path.exists():
        with  open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:

        jersey_predictor = JerseyNumberPredictorParseq(
            paths=paths,
            jersey_cfg=jersey_cfg,
            device=device
        )

        team_classifier = TeamClassifier(
            device=device,
            batch_size=32,
            paths=paths
        )


        tracklet_crops = {}
        player_crops = []
        for track_id, tracklet in tqdm(tracklets.items()):
            full_crops, jerseys, jersey_confs = jersey_predictor.predict(images, tracklet)

            tracklet.pred_attributes['jerseys'] = jerseys.tolist()
            tracklet.pred_attributes['jersey_confs'] = jersey_confs.tolist()

            # Store all crops for team classification
            tracklet_crops[track_id] = full_crops
            roles = tracklet.gt_attributes.get('roles', [])
            for crop_idx, crop in enumerate(full_crops):
                if roles[crop_idx] == "player":
                    player_crops.append(crop)

        team_classifier.fit(player_crops)

        for track_id, tracklet in tqdm(tracklets.items()):
            full_crops = tracklet_crops[track_id]
            team_predictions = team_classifier.predict(full_crops)
            tracklet.pred_attributes['teams'] = team_predictions.tolist()

            if track_id == 16:
                save_crops(full_crops)

        print("team prediction finished")
            


    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(tracklets, f)

    return tracklets



def save_crops(crops):
    import os
    output_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\debug\reid_16"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for i, crop in enumerate(crops):
        if crop.size == 0:
            continue

        filename = f"crop_{i}.jpg"
        file_path = os.path.join(output_dir, filename)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # 4. Save the image
        cv2.imwrite(file_path, crop_bgr)
        
    print(f"Saved {len(crops)} images to {output_dir}")