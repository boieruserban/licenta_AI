import torch.nn as nn
from transformers import CLIPModel

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CLIPClassifier, self).__init__()
        # Always load pretrained CLIP base
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = self.clip_model.vision_model  # use only vision part
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.clip(x).last_hidden_state[:, 0, :]
        features = self.dropout(features)
        return self.fc(features)

def build_model(num_classes):
    model = CLIPClassifier(num_classes)
    return model

def freeze_clip_layers(model, unfreeze_last=4):
    for param in model.clip.parameters():
        param.requires_grad = False

    encoder = model.clip.encoder
    if hasattr(encoder, 'layers'):
        to_unfreeze = encoder.layers[-unfreeze_last:]
        for block in to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
