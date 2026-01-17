"""
Jersey number prediction pipeline.
Based on mkoshkina's jersey-number-pipeline (https://github.com/mkoshkina/jersey-number-pipeline)
"""
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

from .parseq.strhub.data.module import SceneTextDataModule
from .legibility_classifier import LegibilityPredictor
from .pose_cropper import PoseCropper
from .centroid_reid_filter import CentroidReIDFilter


class JerseyNumberPredictorParseq:
    """
    Jersey number detection pipeline.
    
    Returns:
    - Jersey predictions (filtered for legibility)
    - Torso crops for team classification (unfiltered - we want ALL crops for color-based classification)
    """
    
    def __init__(self, paths, jersey_cfg, device='cpu'):
        self.jersey_cfg = jersey_cfg
        self.device = device
        self.debug_dir = Path(jersey_cfg.debug_dir) if jersey_cfg.debug_dir else None
        self.paths = paths

        self.parseq_model = self.load_parseq_model(paths.parseq_model_path)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq_model.hparams.img_size)

        self.legibility_predictor = self.load_legibility_model(paths.legibility_model_path) if jersey_cfg.use_legibility else None
        self.pose_cropper = self.load_pose_cropper_model(paths.vitpose_model_path) if jersey_cfg.use_pose_cropper else None
        self.reid_filter = self.load_reid_filter_model(paths.centroid_reid_path) if jersey_cfg.use_reid_filter else None


    def predict(self, images, tracklet):
        """ Predict jersey numbers for a tracklet """
        num_frames = len(tracklet.frames)
        
        # Extract all crops
        full_crops, torso_crops, indices = self.extract_crops(images, tracklet)
        
        # Keep unfiltered torso crops for team classification
        torso_crops_for_teams = torso_crops.copy()
        indices_for_teams = indices.copy()

        if not full_crops:
            return [], [], np.full(num_frames, np.nan), np.zeros(num_frames)
        
        
        # Stage 1: ReID outlier filtering
        if self.reid_filter and tracklet.embeddings:
            embeddings = [tracklet.embeddings[i] for i in indices]
            [full_crops, torso_crops], indices = self.reid_filter.filter(
                [full_crops, torso_crops], indices, embeddings
            )
                
        # Stage 2: Legibility filtering
        if self.legibility_predictor and full_crops:
            [full_crops, torso_crops], indices = self.legibility_predictor.filter(
                [full_crops, torso_crops], indices
            )

        # Stage 3: Predict jersey numbers on filtered torso crops
        if torso_crops:
            predictions, confidences = self.predict_jersey(torso_crops)
        else:
            predictions, confidences = [], []
        
        jerseys, confs = self.map_to_frames(predictions, confidences, indices, num_frames)

        return torso_crops_for_teams, indices_for_teams, jerseys, confs


    def extract_crops(self, images, tracklet):
        """ Extract full and torso crops for all frames in tracklet """
        full_crops = []
        torso_crops = []
        indices = []

        for i, (frame_idx, bbox) in enumerate(zip(tracklet.frames, tracklet.bboxes)):
            image = cv2.imread(str(images[frame_idx]))
            if image is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            full_crop = image[y1:y2, x1:x2]
            if full_crop.size == 0:
                continue
                
            full_crop_rgb = cv2.cvtColor(full_crop, cv2.COLOR_BGR2RGB)

            # Get torso crop using pose estimation or fallback to heuristic
            if self.pose_cropper:
                torso_crop = self.pose_cropper.get_torso_crop(image, bbox)
                if torso_crop is None:
                    torso_crop = self.simple_torso_crop(full_crop_rgb)
            else:
                torso_crop = self.simple_torso_crop(full_crop_rgb)

            if torso_crop is None or torso_crop.size == 0:
                continue

            full_crops.append(full_crop_rgb)
            torso_crops.append(torso_crop)
            indices.append(i)

        return full_crops, torso_crops, indices
        

    def load_parseq_model(self, parseq_model_path):
        """ Load PARSeq OCR model """
        model = torch.load(parseq_model_path, map_location='cpu', weights_only=False)

        if 'state_dict' in model:
            state_dict = model['state_dict']
        elif 'model' in model:
            state_dict = model['model']
        else:
            state_dict = model

        new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}

        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=False)
        model.load_state_dict(new_state_dict, strict=True)
        model = model.eval().to(self.device)
        
        print("PARSeq model loaded")
        return model
    

    def load_legibility_model(self, legibility_model_path):
        """ Load legibility classifier """
        return LegibilityPredictor(
            model_path=str(legibility_model_path),
            device=self.device,
            threshold=self.jersey_cfg.legibility_threshold,
            arch=self.jersey_cfg.legibility_arch
        )
    
    
    def load_pose_cropper_model(self, vitpose_model_path):
        """ Load ViTPose model for torso cropping """
        # Use local path if available, otherwise HuggingFace
        return PoseCropper(
            device=self.device,
            model_path=str(vitpose_model_path)
        )

    
    def load_reid_filter_model(self, centroid_reid_path):
        """ Load Centroid-ReID model for outlier filtering """
        return CentroidReIDFilter(
            checkpoint_path=centroid_reid_path,
            threshold_std=self.jersey_cfg.reid_threshold_std,
            rounds=5,
            min_samples=3
        )
    

    def map_to_frames(self, predictions, confidences, indices, num_frames):
        """ Map predictions back to all tracklet frames """
        result_map = dict(zip(indices, zip(predictions, confidences)))
        
        jerseys = np.array([result_map.get(i, (np.nan, 0.0))[0] for i in range(num_frames)])
        confs = np.array([result_map.get(i, (np.nan, 0.0))[1] for i in range(num_frames)])
        
        return jerseys, confs
    

    def predict_jersey(self, crops, batch_size=32):
        """ Run PARSeq on crops to predict jersey numbers """
        jerseys = []
        confidences = []
        
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            tensors = [self.img_transform(Image.fromarray(c)) for c in batch]
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                logits = self.parseq_model(batch_tensor)
                probs = logits.softmax(-1)
            
            labels, raw_confs = self.parseq_model.tokenizer.decode(probs)
            
            for label, conf_tensor in zip(labels, raw_confs):
                label = label.strip()
                conf = conf_tensor.mean().item()
                
                if label.isdigit() and 0 <= int(label) <= 99:
                    jerseys.append(int(label))
                    confidences.append(conf)
                else:
                    jerseys.append(np.nan)
                    confidences.append(0.0)
        
        return jerseys, confidences
    

    def simple_torso_crop(self, crop):
        """ Simple heuristic torso crop """
        h = crop.shape[0]
        return crop[int(h * 0.2):int(h * 0.8), :, :]