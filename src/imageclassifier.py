import os
import queue
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from datasets import load_dataset
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, DISABLED, NORMAL

#setup gpu training if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load clip model
base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#load dataset for training
dataset = load_dataset("tukey/human_face_emotions_roboflow")

#extract lables
emotion_classes = list(set([sample["qa"][0]["answer"] for sample in dataset["train"]]))
num_classes = len(emotion_classes)

label_to_index = {label: idx for idx, label in enumerate(emotion_classes)}

#preprocessing
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

#model definition
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

#freezing layers
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

freeze_clip_layers(classifier, unfreeze_last=4)

#gui
root = tk.Tk()
root.title("Emotion Classifier")
root.geometry("900x700")

epochs_var = tk.StringVar(value="8")
split_var = tk.StringVar(value="0.8")
batch_var = tk.StringVar(value="32")
lr_var = tk.StringVar(value="0.00001")
wd_var = tk.StringVar(value="0.0001")

training_in_progress = False
current_image_tk = None

log_queue = queue.Queue()

progress_var = tk.DoubleVar(value=0.0)

top_frame = tk.Frame(root)
top_frame.pack(pady=5, fill=tk.X)

param_frame = tk.LabelFrame(root, text="Hyperparameters")
param_frame.pack(pady=5, fill=tk.X)

middle_frame = tk.Frame(root)
middle_frame.pack(pady=5, fill=tk.BOTH, expand=True)

bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=5, fill=tk.X)

log_text = tk.Text(middle_frame, wrap=tk.WORD, width=70, height=15)
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(middle_frame, command=log_text.yview)
scrollbar.pack(side=tk.LEFT, fill=tk.Y)
log_text.config(yscrollcommand=scrollbar.set)

def log_message(msg):
    log_queue.put(msg)

def display_logs_from_queue():
    while not log_queue.empty():
        msg = log_queue.get_nowait()
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
    root.after(200, display_logs_from_queue)

#buttons
train_button = tk.Button(top_frame, text="Train Model", font=("Arial", 12))
train_button.pack(side=tk.LEFT, padx=5)

save_button = tk.Button(top_frame, text="Save Model", font=("Arial", 12))
save_button.pack(side=tk.LEFT, padx=5)

load_button = tk.Button(top_frame, text="Load trained model", font=("Arial", 12))
load_button.pack(side=tk.LEFT, padx=5)

classify_button = tk.Button(top_frame, text="Classify Image", font=("Arial", 12), state=DISABLED)
classify_button.pack(side=tk.LEFT, padx=5)

train_progress = ttk.Progressbar(top_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var)
train_progress.pack(side=tk.LEFT, padx=10)

#hyperparameters
tk.Label(param_frame, text="Epochs:").pack(side=tk.LEFT, padx=5)
epochs_entry = tk.Entry(param_frame, textvariable=epochs_var, width=5)
epochs_entry.pack(side=tk.LEFT, padx=5)

tk.Label(param_frame, text="Train/Val Split:").pack(side=tk.LEFT, padx=5)
split_entry = tk.Entry(param_frame, textvariable=split_var, width=5)
split_entry.pack(side=tk.LEFT, padx=5)

tk.Label(param_frame, text="Batch Size:").pack(side=tk.LEFT, padx=5)
batch_entry = tk.Entry(param_frame, textvariable=batch_var, width=5)
batch_entry.pack(side=tk.LEFT, padx=5)

tk.Label(param_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
lr_entry = tk.Entry(param_frame, textvariable=lr_var, width=8)
lr_entry.pack(side=tk.LEFT, padx=5)

tk.Label(param_frame, text="Weight Decay:").pack(side=tk.LEFT, padx=5)
wd_entry = tk.Entry(param_frame, textvariable=wd_var, width=8)
wd_entry.pack(side=tk.LEFT, padx=5)

#image preview and classification
image_label = tk.Label(bottom_frame)
image_label.pack(side=tk.LEFT, padx=10)

prob_text = tk.Text(bottom_frame, width=40, height=8)
prob_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#data loader
def build_dataloaders_and_optimizer():
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

    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    return {
        "epochs": num_epochs,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

def validate_model(val_loader, criterion):
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_accuracy = (val_correct / val_total) * 100 if val_total > 0 else 0
    return avg_val_loss, val_accuracy

def train_thread(epochs, train_loader, val_loader, optimizer, scheduler):
    global training_in_progress

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

        # End of epoch
        avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = (epoch_correct / epoch_total) * 100 if epoch_total > 0 else 0

        val_loss, val_acc = validate_model(val_loader, criterion)

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
        train_button.config(state=NORMAL)
        classify_button.config(state=NORMAL)

    root.after(0, re_enable_buttons)

def train_model():
    global training_in_progress
    if training_in_progress:
        return

    result = build_dataloaders_and_optimizer()
    if not result:
        return

    training_in_progress = True
    train_button.config(state=DISABLED)
    classify_button.config(state=DISABLED)
    log_text.delete("1.0", tk.END)
    progress_var.set(0.0)

    t = threading.Thread(
        target=train_thread,
        args=(
            result["epochs"],
            result["train_loader"],
            result["val_loader"],
            result["optimizer"],
            result["scheduler"]
        ),
        daemon=True
    )
    t.start()

train_button.config(command=train_model)

def save_trained_model():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".pth",
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    if file_path:
        torch.save(classifier.state_dict(), file_path)
        messagebox.showinfo("Saved", f"Model saved to {file_path}")

save_button.config(command=save_trained_model)

def load_best_model():
    if not os.path.exists("emotion_classifier_best.pth"):
        messagebox.showwarning("Warning", "No best model found. Train first or load a saved model.")
        return
    state_dict = torch.load("emotion_classifier_best.pth", map_location=device, weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.eval()

#loading pretrained model
def load_user_model():
    file_path = filedialog.askopenfilename(
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    try:
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{file_path}")
        classify_button.config(state=NORMAL)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model: {e}")

load_button.config(command=load_user_model)

#image classification function
def classify_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")

    #resizing
    display_img = img.resize((200, 200), resample=Image.Resampling.LANCZOS)
    global current_image_tk
    current_image_tk = ImageTk.PhotoImage(display_img)
    image_label.config(image=current_image_tk)

    #transformation
    transformed_img = data_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(transformed_img)
        probs = nn.Softmax(dim=1)(outputs).squeeze().cpu().numpy()

    prob_text.delete("1.0", tk.END)
    sorted_indices = probs.argsort()[::-1]
    for idx in sorted_indices:
        class_label = emotion_classes[idx]
        class_prob = probs[idx] * 100
        prob_text.insert(tk.END, f"{class_label}: {class_prob:.2f}%\n")

classify_button.config(command=classify_image)

root.after(200, display_logs_from_queue)

root.mainloop()
