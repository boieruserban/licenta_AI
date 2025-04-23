import tkinter as tk
from tkinter import ttk, filedialog, messagebox, DISABLED, NORMAL
from PIL import Image, ImageTk
import torch
import torch.nn as nn

import numpy as np

from features.train import train_model
from features.realtime import real_time_detection
from features.data import build_data_transforms

def create_gui(
    root,
    classifier,
    device,
    full_dataset,
    emotion_classes,
    log_queue,
    progress_var,
    train_button_callback,
    save_button_callback,
    load_button_callback,
    classify_button_callback,
    realtime_button_callback,
    epochs_var,
    split_var,
    batch_var,
    lr_var,
    wd_var
):

    root.title("Emotion Classifier")
    root.geometry("900x700")

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
            message = log_queue.get_nowait()
            log_text.insert(tk.END, message + "\n")
            log_text.see(tk.END)
        root.after(200, display_logs_from_queue)

    train_button = tk.Button(top_frame, text="Train Model", font=("Arial", 12),
                             command=train_button_callback)
    train_button.pack(side=tk.LEFT, padx=5)

    save_button = tk.Button(top_frame, text="Save Model", font=("Arial", 12),
                            command=save_button_callback)
    save_button.pack(side=tk.LEFT, padx=5)

    load_button = tk.Button(top_frame, text="Load trained model", font=("Arial", 12),
                            command=load_button_callback)
    load_button.pack(side=tk.LEFT, padx=5)

    classify_button = tk.Button(top_frame, text="Classify Image", font=("Arial", 12),
                                command=classify_button_callback, state=DISABLED)
    classify_button.pack(side=tk.LEFT, padx=5)

    realtime_button = tk.Button(top_frame, text="Real-time Detection", font=("Arial", 12),
                                command=realtime_button_callback, state=DISABLED)
    realtime_button.pack(side=tk.LEFT, padx=5)

    train_progress = ttk.Progressbar(top_frame, orient="horizontal", length=200,
                                     mode="determinate", variable=progress_var)
    train_progress.pack(side=tk.LEFT, padx=10)

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

    image_label = tk.Label(bottom_frame)
    image_label.pack(side=tk.LEFT, padx=10)

    prob_text = tk.Text(bottom_frame, width=40, height=8)
    prob_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    root.after(200, display_logs_from_queue)

    def classify_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        img = Image.open(file_path).convert("RGB")
        display_img = img.resize((200, 200), resample=Image.Resampling.LANCZOS)
        current_image_tk = ImageTk.PhotoImage(display_img)
        image_label.config(image=current_image_tk)
        image_label.image = current_image_tk  # store reference

        transform_for_inference = build_data_transforms()
        transformed_img = transform_for_inference(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classifier(transformed_img)
            probs = nn.Softmax(dim=1)(outputs).squeeze().cpu().numpy()

        prob_text.delete("1.0", tk.END)
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            class_label = emotion_classes[idx]
            class_prob = probs[idx] * 100
            prob_text.insert(tk.END, f"{class_label}: {class_prob:.2f}%\n")

    classify_button.config(command=classify_image)

    return {
        "train_button": train_button,
        "classify_button": classify_button,
        "realtime_button": realtime_button,
        "log_text": log_text,
        "log_message": log_message,
        "image_label": image_label,
        "prob_text": prob_text
    }
