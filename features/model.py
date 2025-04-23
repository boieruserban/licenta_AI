import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPClassifier(nn.Module):

    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model.vision_model
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.clip(x).last_hidden_state[:, 0, :]
        features = self.dropout(features)
        return self.fc(features)

def freeze_clip_layers(model, unfreeze_last=4):

    for param in model.clip.parameters():
        param.requires_grad = False

    encoder = model.clip.encoder
    if hasattr(encoder, 'layers'):
        to_unfreeze = encoder.layers[-unfreeze_last:] if unfreeze_last > 0 else []
        for block in to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
    else:
        for param in model.clip.parameters():
            param.requires_grad = True
