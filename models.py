# models.py: Defines the multi-head NutriVisionNetMultiHead model.
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cuda", fine_tune_clip_image=False):
        super(NutriVisionNetMultiHead, self).__init__()
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        if not fine_tune_clip_image:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for param in self.blip_model.parameters():
            param.requires_grad = False
        fusion_dim = 1024
        hidden_dim = 512
        self.food_nutrition_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, food_nutrition_dim)
        )
        self.fv_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fv_dim)
        )
        self.fastfood_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fastfood_dim)
        )
    
    def forward(self, image, source):
        blip_inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.blip_model.generate(**blip_inputs, max_length=50, num_beams=5)
        caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        clip_image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_embeds = self.clip_model.get_image_features(**clip_image_inputs)
        clip_text_inputs = self.clip_processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
        text_embeds = self.clip_model.get_text_features(**clip_text_inputs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        fused = torch.cat([image_embeds, text_embeds], dim=-1)
        if source == "food_nutrition":
            pred = self.food_nutrition_head(fused)
        elif source == "fv":
            pred = self.fv_head(fused)
        elif source == "fastfood":
            pred = self.fastfood_head(fused)
        else:
            raise ValueError(f"Unknown source: {source}")
        return pred, caption
