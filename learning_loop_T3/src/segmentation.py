import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image


class SAMSegmenter:
    def __init__(
        self,
        checkpoint_path="sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device=None
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading SAM {model_type} model on {self.device}...")

        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(self.device)
            self.sam.eval()

            self.predictor = SamPredictor(self.sam)

            print("SAM loaded successfully (low-memory mode).")

        except Exception as e:
            print(f"SAM failed: {e}")
            self.predictor = None

    def segment(self, image):

        if self.predictor is None:
            return image

        try:
            image_np = np.array(image)

            self.predictor.set_image(image_np)

            height, width = image_np.shape[:2]

            # Use center point only → VERY low memory
            input_point = np.array([[width // 2, height // 2]])
            input_label = np.array([1])

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            mask = masks[0]

            segmented = np.zeros_like(image_np)
            segmented[mask] = image_np[mask]

            return Image.fromarray(segmented)

        except Exception as e:
            print(f"Segmentation error: {e}")
            return image