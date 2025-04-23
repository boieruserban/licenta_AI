import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import threading
from tkinter import DISABLED, NORMAL

def build_dataloaders_and_optimizer(
    full_dataset,
    classifier,
    split_var,
    epochs_var,
    batch_var,
    lr_var,
    wd_var,
    messagebox
):

    try:
        train_val_ratio = float(split_var.get())
        num_epochs = int(epochs_var.get())
        batch_size = int(batch_var.get())
        learning_rate = float(lr_var.get())
        weight_decay = float(wd_var.get())
    except ValueError:
        messagebox.showerror("Error", "Enter valid numeric hyperparameter values.")
        return None

    if not 0.0 < train_val_ratio < 1.0:
        messagebox.showerror("Error", "Train/Val split must be between 0 and 1.")
        return None

    train_size = int(train_val_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    return {
        "epochs": num_epochs,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

def validate_model(val_loader, classifier, device):

    criterion = nn.CrossEntropyLoss()
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

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_accuracy = (val_correct / val_total) * 100 if val_total > 0 else 0
    return avg_val_loss, val_accuracy

def train_thread(
    epochs, train_loader, val_loader, optimizer, scheduler,
    classifier, device, progress_var, log_message,
    train_button, classify_button, realtime_button, root
):

    training_in_progress = True
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
            progress = (current_step / total_steps) * 100
            progress_var.set(progress)

        avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = (epoch_correct / epoch_total) * 100 if epoch_total > 0 else 0

        val_loss, val_acc = validate_model(val_loader, classifier, device)
        scheduler.step()

        msg = (f"Epoch {epoch+1}/{epochs}:\n"
               f"  - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
               f"  - Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%\n")
        log_message(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), "emotion_classifier_best.pth")
            log_message("  -> Model updated\n")

    log_message("Training Completed Successfully!")
    training_in_progress = False

    def re_enable_buttons():
        train_button.config(state="normal")
        classify_button.config(state="normal")
        realtime_button.config(state="normal")

    root.after(0, re_enable_buttons)

def train_model(
    full_dataset,
    classifier, device,
    split_var, epochs_var, batch_var, lr_var, wd_var,
    train_button, classify_button, realtime_button,
    log_text, progress_var, log_message, root, messagebox
):

    from .train import build_dataloaders_and_optimizer
    result = build_dataloaders_and_optimizer(
        full_dataset, classifier,
        split_var, epochs_var, batch_var, lr_var, wd_var, messagebox
    )
    if not result:
        return

    train_button.config(state=DISABLED)
    classify_button.config(state=DISABLED)
    realtime_button.config(state=DISABLED)
    log_text.delete("1.0", "end")
    progress_var.set(0.0)

    # Start the training thread
    t = threading.Thread(
        target=train_thread,
        args=(
            result["epochs"],
            result["train_loader"],
            result["val_loader"],
            result["optimizer"],
            result["scheduler"],
            classifier,
            device,
            progress_var,
            log_message,
            train_button,
            classify_button,
            realtime_button,
            root
        ),
        daemon=True
    )
    t.start()
