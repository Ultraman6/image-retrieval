import numpy as np
import torch
from torchvision import transforms

class DinoV2:
    dim = 768
    def __init__(self, model_size, model_path):
        # Load dinov2 model
        model_name_dict = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
            "largest": "dinov2_vitg14",
        }
        model_name = model_name_dict[model_size]
        model_folder = (
            "facebookresearch/dinov2" if model_path is None else model_path
        )
        model_source = "github" if model_path is None else "local"
        self.model = torch.hub.load(
            model_folder,
            model_name,
            source=model_source,
        )
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def calculate_distance(self, query_feature, database_features):
        cosine_distances = [
            np.dot(query_feature, feature)
            / (np.linalg.norm(query_feature) * np.linalg.norm(feature))
            for feature in database_features
        ]
        return cosine_distances

    def extract_single_image_feature(self, image):
        net_input = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = self.model(net_input).squeeze().numpy()
        return feature
