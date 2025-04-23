import os
import queue
import tkinter as tk
import torch

from features.model import CLIPModel, CLIPProcessor, CLIPClassifier, freeze_clip_layers
from features.data import load_emotion_dataset
from features.gui import create_gui
from features.train import train_model
from features.realtime import real_time_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP
base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset
full_dataset, emotion_classes, label_to_index = load_emotion_dataset()

# Classifier
classifier = CLIPClassifier(base_clip_model, len(emotion_classes)).to(device)
freeze_clip_layers(classifier, unfreeze_last=4)

# Tkinter init
root = tk.Tk()

epochs_var = tk.StringVar(value="8")
split_var = tk.StringVar(value="0.8")
batch_var = tk.StringVar(value="32")
lr_var = tk.StringVar(value="0.00001")
wd_var = tk.StringVar(value="0.0001")

progress_var = tk.DoubleVar(value=0.0)
log_queue = queue.Queue()

# Buttons

def on_train_button_click():

    from features.train import train_model
    train_model(
        full_dataset=full_dataset,
        classifier=classifier,
        device=device,
        split_var=split_var,
        epochs_var=epochs_var,
        batch_var=batch_var,
        lr_var=lr_var,
        wd_var=wd_var,
        train_button=gui_elements["train_button"],
        classify_button=gui_elements["classify_button"],
        realtime_button=gui_elements["realtime_button"],
        log_text=gui_elements["log_text"],
        progress_var=progress_var,
        log_message=gui_elements["log_message"],
        root=root,
        messagebox=tk.messagebox
    )

def on_save_model_click():

    file_path = tk.filedialog.asksaveasfilename(
        defaultextension=".pth",
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    if file_path:
        torch.save(classifier.state_dict(), file_path)
        tk.messagebox.showinfo("Saved", f"Model saved to {file_path}")

def on_load_model_click():

    file_path = tk.filedialog.askopenfilename(
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    if not file_path:
        return

    try:
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        classifier.load_state_dict(state_dict)
        classifier.to(device)
        classifier.eval()
        tk.messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{file_path}")
        gui_elements["classify_button"].config(state="normal")
        gui_elements["realtime_button"].config(state="normal")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Could not load model: {e}")

def on_classify_button_click():

    pass

def on_realtime_button_click():

    from features.data import build_data_transforms
    transform_for_inference = build_data_transforms()
    real_time_detection(
        classifier=classifier,
        data_transform=transform_for_inference,
        emotion_classes=emotion_classes,
        device=device
    )

# GUI
gui_elements = create_gui(
    root=root,
    classifier=classifier,
    device=device,
    full_dataset=full_dataset,
    emotion_classes=emotion_classes,
    log_queue=log_queue,
    progress_var=progress_var,
    train_button_callback=on_train_button_click,
    save_button_callback=on_save_model_click,
    load_button_callback=on_load_model_click,
    classify_button_callback=on_classify_button_click,
    realtime_button_callback=on_realtime_button_click,
    epochs_var=epochs_var,
    split_var=split_var,
    batch_var=batch_var,
    lr_var=lr_var,
    wd_var=wd_var
)

root.mainloop()
