import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

def build_data_transforms():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

class EmotionDataset(Dataset):

    def __init__(self, hf_dataset, label_to_index, transform):
        self.data = hf_dataset
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label_str = self.data[idx]["qa"][0]["answer"]
        label = self.label_to_index[label_str]
        image = self.transform(image)
        return image, label

def load_emotion_dataset():

    dataset = load_dataset("tukey/human_face_emotions_roboflow")
    emotion_classes = list(set([sample["qa"][0]["answer"] for sample in dataset["train"]]))
    label_to_index = {label: idx for idx, label in enumerate(emotion_classes)}

    transform = build_data_transforms()
    full_dataset = EmotionDataset(dataset["train"], label_to_index, transform)

    return full_dataset, emotion_classes, label_to_index
