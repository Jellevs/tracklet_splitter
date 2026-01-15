### ADD ONLY JERSEY PREDICTION ON GT ROLES FROM GK AND PLAYERS
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
    """ Mkoshkina's jersey number detection pipeline (https://github.com/mkoshkina/jersey-number-pipeline) """
    def __init__(self, paths, jersey_cfg, device='cpu'):
        self.jersey_cfg = jersey_cfg
        self.device = device
        self.debug_dir = Path(jersey_cfg.debug_dir) if jersey_cfg.debug_dir else None

        self.parseq_model = self.load_parseq_model(paths.parseq_model_path)
        self.img_transform = SceneTextDataModule.get_transform(self.parseq_model.hparams.img_size)

        self.legibility_predictor = self.load_legibility_model(paths.legibility_model_path) if jersey_cfg.use_legibility else None
        self.pose_cropper = self.load_pose_cropper_model() if jersey_cfg.use_pose_cropper else None
        self.reid_filter = self.load_reid_filter_model(paths.centroid_reid_path) if jersey_cfg.use_reid_filter else None


    def predict(self, images, tracklet):
        """Predict jersey numbers for a tracklet"""
        num_frames = len(tracklet.frames)
        
        full_crops, torso_crops, indices = self.extract_crops(images, tracklet)

        if not full_crops:
            return np.full(num_frames, np.nan), np.zeros(num_frames)
        

        # Stage 1: ReID outlier filtering
        if self.reid_filter and tracklet.embeddings:
            embeddings = [tracklet.embeddings[i] for i in indices]
            
            [full_crops, torso_crops], indices = self.reid_filter.filter(
                [full_crops, torso_crops], indices, embeddings
            )

        full_crops_teams_with_outlier_filter = full_crops.copy()        

                
        # Stage 2: Legibility filtering
        if self.legibility_predictor and full_crops:
            legibility_flags, legibility_confs = self.legibility_predictor.predict_batch(full_crops)

            # Enable for legibility debug
            # if self.debug_dir and self.jersey_cfg.debug_tracklet_id == tracklet.parent_id:
            #     self.save_legibility_debug(tracklet, full_crops,  indices, legibility_flags)

            [full_crops, torso_crops], indices = self.legibility_predictor.filter(
                [full_crops, torso_crops], indices
            )

        # Stage 3: Predict on remaining torso crops
        if torso_crops:
            predictions, confidences = self.predict_jersey(torso_crops)
        else:
            predictions, confidences = [], []
        
        jerseys, confs = self.map_to_frames(predictions, confidences, indices, num_frames)

        return full_crops_teams_with_outlier_filter, jerseys, confs


    def extract_crops(self, images, tracklet):
        """Extract player crops"""

        full_crops = []
        torso_crops = []
        indices = []

        for i, (frame_idx, bbox) in enumerate(zip(tracklet.frames, tracklet.bboxes)):

            # Only extract crops from goalkeepers and players
            # if tracklet.gt_attributes['roles'][i] == "referee":
            #     continue

            image = cv2.imread(str(images[frame_idx]))

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            full_crop = image[y1:y2, x1:x2]
            full_crop_rgb = cv2.cvtColor(full_crop, cv2.COLOR_BGR2RGB)

            if self.pose_cropper:
                torso_crop = self.pose_cropper.get_torso_crop(image, bbox)
                if torso_crop is None:
                    torso_crop  = self.simple_torso_crop(full_crop_rgb)
            else:
                torso_crop = self.simple_torso_crop(full_crop_rgb)

            if torso_crop is None or torso_crop.size == 0:
                continue

            full_crops.append(full_crop_rgb)
            torso_crops.append(torso_crop)
            indices.append(i)

        return full_crops, torso_crops, indices
        

    def load_parseq_model(self, parseq_model_path):
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
        return LegibilityPredictor(
            model_path = str(legibility_model_path),
            device = self.device,
            threshold = self.jersey_cfg.legibility_threshold,
            arch = self.jersey_cfg.legibility_arch
        )
    
    
    def load_pose_cropper_model(self):
        return PoseCropper(
            device=self.device,
        )

    
    def load_reid_filter_model(self, centroid_reid_path):
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
    

    def predict_jersey(self, crops, batch_size = 32):
        """ Run PARSeq on crops """

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
        """ Simple heuristic: middle 60% vertically """
        h = crop.shape[0]
        return crop[int(h * 0.2):int(h * 0.8), :, :]


    def save_crops(self, crops):
        import os
        output_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\debug\reid"
        
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


    def save_reid_crops(self, crops):
        import os
        output_dir = r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\debug\reid_filtered_centroids_real"
        
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


    def save_legibility_debug(self, tracklet, full_crops, indices, legibility_flags):
        """ Save crops with green (legible) or red (illegible) borders """
        
        if not self.debug_dir or self.jersey_cfg.debug_tracklet_id != tracklet.track_id:
            return
        
        debug_path = self.debug_dir / f"tracklet_{tracklet.track_id}_legibility"
        debug_path.mkdir(parents=True, exist_ok=True)

        print(f"   Total crops: {len(full_crops)}, Legible: {sum(legibility_flags)}")
        
        for idx, (crop, frame_idx, is_legible) in enumerate(zip(full_crops, indices, legibility_flags)):
            # Create a copy with border
            crop_with_border = crop.copy()
            
            # Add colored border (green=legible, red=illegible)
            border_color = (0, 255, 0) if is_legible else (255, 0, 0)  # RGB
            border_thickness = 5
            
            h, w = crop_with_border.shape[:2]
            crop_with_border = cv2.copyMakeBorder(
                crop_with_border,
                border_thickness, border_thickness,
                border_thickness, border_thickness,
                cv2.BORDER_CONSTANT,
                value=border_color
            )
            
            # Save with frame number
            actual_frame = tracklet.frames[frame_idx]
            filename = f"frame_{actual_frame:04d}_idx_{idx:03d}_{'legible' if is_legible else 'illegible'}.png"
            
            # Convert RGB to BGR for saving
            crop_bgr = cv2.cvtColor(crop_with_border, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_path / filename), crop_bgr)
        
        print(f"âœ… Saved {len(full_crops)} debug images")