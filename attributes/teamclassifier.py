from transformers import AutoProcessor, SiglipVisionModel


class TeamClassifier:
    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            
        )