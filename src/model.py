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

classifier = CLIPClassifier(base_clip_model, num_classes).to(device)