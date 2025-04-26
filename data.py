import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset

dataset = load_dataset("tukey/human_face_emotions_roboflow")

emotion_classes = sorted(list(set([sample["qa"][0]["answer"] for sample in dataset["train"]])))
label_to_index = {label: idx for idx, label in enumerate(emotion_classes)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

num_classes = len(emotion_classes)

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class EmotionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = label_to_index[self.data[idx]["qa"][0]["answer"]]
        image = data_transform(image)
        return image, label

full_dataset = EmotionDataset(dataset["train"])

def build_dataloaders(train_val_ratio, batch_size):
    """Build and return train and validation dataloaders."""
    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
