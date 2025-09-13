import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data import build_dataloaders, emotion_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_confusion_matrix_report(input_model_or_preds, val_loader, log_fn, mode="Trained Model"):
    log_fn(f"\n[Confusion Matrix - {mode}]\n")

    if isinstance(input_model_or_preds, torch.nn.Module):
        model = input_model_or_preds
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    else:
        all_preds, all_labels = input_model_or_preds

    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=emotion_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=emotion_classes,
                yticklabels=emotion_classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {mode}")
    plt.tight_layout()
    plt.show()

    log_fn(cr)
