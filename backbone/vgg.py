import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA


class VGGNet(object):
    dim = 25088
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load VGG16 pre-trained on ImageNet
        self.model_vgg = models.vgg16(pretrained=True).features
        self.model_vgg.to(self.device).eval()

        # Define the image transformations to match VGG16 preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to [C, H, W] format
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_single_image_feature(self, img):
        # Load the image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass through VGG16 to extract features
        with torch.no_grad():
            feat = self.model_vgg(img_tensor)

        # Flatten the feature map to a 1D vector and normalize
        feat = feat.view(feat.size(0), -1)  # Flatten the feature map
        norm_feat = feat / LA.norm(feat.cpu().numpy())  # Normalize the feature vector

        # Convert to list and return
        norm_feat = norm_feat[0].cpu().numpy().tolist()
        return norm_feat
