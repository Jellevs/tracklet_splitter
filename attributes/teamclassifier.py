from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
import supervision as sv
import torch
from tqdm import tqdm
import numpy as np


def create_batches(sequence, batch_size):
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    def __init__(self, device, batch_size, paths):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            pretrained_model_name_or_path=paths.siglip_model_path
            ).to(device)
        self.processor = AutoProcessor.from_pretrained(paths.siglip_model_path)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        self.is_fitted = False
        

    def extract_features(self, crops):
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches):
                inputs = self.processor(
                    images=batch, return_tensors='pt').to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)
    

    def fit(self, crops):
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self.is_fitted = True


    def predict(self, crops):
        if len(crops) == 0:
            return np.array([])
        
        data = self.extract_features(crops)
        projections  = self.reducer.transform(data)
        return self.cluster_model.predict(projections)