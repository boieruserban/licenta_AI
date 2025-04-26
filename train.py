import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    best_val_loss = float("inf")
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

        log_fn(f"Epoch {epoch+1}/{epochs}:")
        log_fn(f"  - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        log_fn(f"  - Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), "emotion_classifier_best.pth")
            log_fn("  -> Model updated\n")

    log_fn("Training Completed Successfully!")
    button_states()
