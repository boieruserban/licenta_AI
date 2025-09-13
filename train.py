import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import threading
from transformers import CLIPProcessor, CLIPModel
from data import build_dataloaders, emotion_classes
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

@torch.no_grad()
def zero_shot_eval(val_loader):
    texts = [f"a photo of a {label} face" for label in emotion_classes]
    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**text_inputs)
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device)
        image_features = clip_model.get_image_features(pixel_values=images)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.T
        preds = torch.argmax(logits_per_image, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

        correct += (preds.cpu() == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return acc, all_preds, all_labels

def validate_model(classifier, val_loader, criterion):
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (val_correct / val_total) * 100
    return avg_val_loss, val_accuracy

def train_thread_fn(classifier, epochs, train_loader, val_loader, optimizer, scheduler,
                    log_fn, progress_var, button_states):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    total_steps = epochs * len(train_loader)
    current_step = 0

    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)

            current_step += 1
            progress_var.set((current_step / total_steps) * 100)

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = (epoch_correct / epoch_total) * 100
        val_loss, val_acc = validate_model(classifier, val_loader, criterion)
        scheduler.step()

        log_fn(f"Epoch {epoch+1}/{epochs}:\n  - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%\n  - Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), "emotion_classifier_best.pth")
            log_fn("  -> Model updated (best validation accuracy)\n")

    log_fn("Training Completed Successfully!\n")

    log_fn("Running Zero-Shot Evaluation...")
    zshot_acc, zshot_preds, zshot_labels = zero_shot_eval(val_loader)
    log_fn(f"Zero-Shot Accuracy on Validation: {zshot_acc:.2f}%\n")

    log_fn("Generating Confusion Matrices...")
    classifier.eval()
    evaluate.compute_confusion_matrix_report(classifier, val_loader, log_fn, mode="Trained Model")
    evaluate.compute_confusion_matrix_report((zshot_preds, zshot_labels), val_loader, log_fn, mode="Zero-Shot")

    button_states()
