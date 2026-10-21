import torch
from torchvision import transforms
from PIL import Image


class DINOv2Extractor:

    def __init__(self, model_path="dinov2_vitb14_pretrain.pth", device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading DINOv2 model from local checkpoint on {self.device}...")

        # Load architecture
        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            pretrained=False
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print("DINOv2 loaded successfully.")

        # Transform ONLY if input is PIL
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])


    def extract_features(self, image):

        # Case 1: PIL Image
        if isinstance(image, Image.Image):

            image_tensor = self.transform(image)

        # Case 2: already tensor
        elif isinstance(image, torch.Tensor):

            image_tensor = image

        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Ensure correct shape: [C,H,W]
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        # Add batch dimension → [1,C,H,W]
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.squeeze().cpu().numpy()