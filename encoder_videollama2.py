import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import AutoImageProcessor, AutoModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2_feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-small',
                                                                          torch_dtype=torch.float16,
                                                                          low_cpu_mem_usage=True)
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-small', torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True)
dinov2_model.eval()
dinov2_model.to(device)

from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
)

def extract_dinov2_features(pixel_values):
    from torchvision.transforms.functional import to_pil_image

    images_pil = [to_pil_image(img) for img in pixel_values.cpu()]

    inputs = dinov2_feature_extractor(images=images_pil, return_tensors="pt")

    inputs = inputs.to(device, dtype=next(dinov2_model.parameters()).dtype)

    with torch.no_grad():
        outputs = dinov2_model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)

    dinov2_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len_dino, hidden_size_dino]
    dinov2_attentions = outputs.attentions #[num_layers, batch_size, num_heads, seq_len_dino, seq_len_dino]
    # print(len(dinov2_attentions))
    return dinov2_hidden_states, dinov2_attentions


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            

            def get_grid_size(num_patches):
                factors = []
                for i in range(1, int(math.sqrt(num_patches)) + 1):
                    if num_patches % i == 0:
                        factors.append((i, num_patches // i))
                # Find the pair with the minimal difference
                factors.sort(key=lambda x: abs(x[0] - x[1]))
                return factors[0]

            dinov2_hidden_states, dinov2_attentions = extract_dinov2_features(images.to(device=self.device, dtype=self.dtype))

      
            last_layer_attentions = dinov2_attentions[-2]  # [batch_size, num_heads, seq_len, seq_len]

            dinov2_avg_attn = last_layer_attentions.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            dinov2_avg_attn = dinov2_avg_attn[:, 1:, 1:] 
            saliency_map = dinov2_avg_attn.sum(dim=1)
            print(saliency_map.shape)

            batch_size = saliency_map.shape[0]
            num_patches_siglip = image_features.shape[1]
            num_patches_dino = saliency_map.shape[1]

            grid_size_siglip_height, grid_size_siglip_width = get_grid_size(num_patches_siglip)
            grid_size_dinov2_height, grid_size_dinov2_width = get_grid_size(num_patches_dino)

            dinov2_saliency_map_2d = saliency_map.view(batch_size, grid_size_dinov2_height, grid_size_dinov2_width)

            dinov2_saliency_map_upsampled = F.interpolate(
                dinov2_saliency_map_2d.unsqueeze(1),
                size=(grid_size_siglip_height, grid_size_siglip_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            saliency_map = dinov2_saliency_map_upsampled.view(batch_size, -1)

            saliency_map = torch.sigmoid(saliency_map)
    
            saliency_expanded = saliency_map.unsqueeze(-1)
            #.repeat(1, 1, frame_features.shape[-1])  # [batch_size, num_patches, feature_dim]
            print(f"saliency_map: {saliency_expanded.shape}")

            # Fuse features by concatenating along the feature dimension
            image_features = image_features * saliency_expanded


        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if  'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    return vision_tower
