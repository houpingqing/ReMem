import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPDoubleTower(nn.Module):
    def __init__(self, model_name, delay_load=False):
        super().__init__()
        self.model_name = model_name
        self.delay_load = delay_load
        if not delay_load:
            self.load_model()

    def load_model(self):
        self.clip = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.device = next(self.clip.parameters()).device

    def get_pooled_features(self, question, frames):
        """
        question: str or None
        frames: torch.Tensor [T, C, H, W] or list of PIL images or None
        return: text_pooled [1, D] or None, image_pooled [T, D] or None
        """
        if self.delay_load and not hasattr(self, "clip"):
            self.load_model()

        text_pooled = None
        image_pooled = None

        # Text feature
        if question is not None:
            inputs = self.processor(text=question, images=None, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k.startswith("input")}
            text_outputs = self.clip.get_text_features(**text_inputs)
            text_pooled = F.normalize(text_outputs, dim=-1)

        # Image feature
        if frames is not None:
            if isinstance(frames, torch.Tensor):
                images = [F.to_pil_image(frame.cpu()) for frame in frames]
            else:
                images = frames
            image_inputs = self.processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            image_outputs = self.clip.get_image_features(**image_inputs)
            image_pooled = F.normalize(image_outputs, dim=-1)

        return text_pooled, image_pooled