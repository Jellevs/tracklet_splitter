from tqdm import tqdm
import pickle

from .jersey_number.jersey_number_predictor_parseq import JerseyNumberPredictorParseq


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

        for track_id, tracklet in tqdm(tracklets.items()):
            jerseys, jersey_confs = jersey_predictor.predict(images, tracklet)

            tracklet.pred_attributes['jerseys'] = jerseys.tolist()
            tracklet.pred_attributes['jersey_confs'] = jersey_confs.tolist()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(tracklets, f)

    return tracklets



