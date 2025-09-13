import tkinter as tk
from tkinter import ttk, filedialog, messagebox, DISABLED, NORMAL
import threading
import torch
from PIL import Image, ImageTk
import queue
import numpy as np
import torch.nn as nn

from data import build_dataloaders, emotion_classes, data_transform, num_classes
from train import train_thread_fn, zero_shot_eval
from realtime import real_time_detection
from model import build_model, freeze_clip_layers
from evaluate import compute_confusion_matrix_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GUI:
    def __init__(self, root, classifier):
        self.root = root
        self.classifier = classifier
        self.training_in_progress = False
        self.current_image_tk = None
        self.log_queue = queue.Queue()

        self.epochs_var = tk.StringVar(value="8")
        self.split_var = tk.StringVar(value="0.8")
        self.batch_var = tk.StringVar(value="32")
        self.lr_var = tk.StringVar(value="0.00001")
        self.wd_var = tk.StringVar(value="0.0001")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.build_gui()

    def build_gui(self):
        self.root.title("Emotion Classifier")
        self.root.geometry("900x700")

        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5, fill=tk.X)

        param_frame = tk.LabelFrame(self.root, text="Hyperparameters")
        param_frame.pack(pady=5, fill=tk.X)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=5, fill=tk.X)

        self.train_button = tk.Button(top_frame, text="Train Model", font=("Arial", 12), command=self.train_model)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(top_frame, text="Save Model", font=("Arial", 12), command=self.save_model)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(top_frame, text="Load trained model", font=("Arial", 12), command=self.load_model)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.classify_button = tk.Button(top_frame, text="Classify Image", font=("Arial", 12), state=DISABLED, command=self.classify_image)
        self.classify_button.pack(side=tk.LEFT, padx=5)

        self.realtime_button = tk.Button(top_frame, text="Real-time Detection", font=("Arial", 12), state=DISABLED, command=self.realtime_detection)
        self.realtime_button.pack(side=tk.LEFT, padx=5)

        self.eval_button = tk.Button(top_frame, text="Show Confusion Matrix", font=("Arial", 12), state=DISABLED, command=self.show_confusion_matrix)
        self.eval_button.pack(side=tk.LEFT, padx=5)

        self.zshot_button = tk.Button(top_frame, text="Zero-Shot Eval", font=("Arial", 12), command=self.show_zero_shot_eval)
        self.zshot_button.pack(side=tk.LEFT, padx=5)

        self.train_progress = ttk.Progressbar(top_frame, orient="horizontal", length=200,
                                              mode="determinate", variable=self.progress_var)
        self.train_progress.pack(side=tk.LEFT, padx=10)

        tk.Label(param_frame, text="Epochs:").pack(side=tk.LEFT, padx=5)
        tk.Entry(param_frame, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(param_frame, text="Train/Val Split:").pack(side=tk.LEFT, padx=5)
        tk.Entry(param_frame, textvariable=self.split_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(param_frame, text="Batch Size:").pack(side=tk.LEFT, padx=5)
        tk.Entry(param_frame, textvariable=self.batch_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(param_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
        tk.Entry(param_frame, textvariable=self.lr_var, width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(param_frame, text="Weight Decay:").pack(side=tk.LEFT, padx=5)
        tk.Entry(param_frame, textvariable=self.wd_var, width=8).pack(side=tk.LEFT, padx=5)

        self.log_text = tk.Text(middle_frame, wrap=tk.WORD, width=70, height=15)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(middle_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.image_label = tk.Label(bottom_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)

        self.prob_text = tk.Text(bottom_frame, width=40, height=8)
        self.prob_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root.after(200, self.display_logs_from_queue)

    def log_message(self, msg):
        self.log_queue.put(msg)

    def display_logs_from_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.root.after(200, self.display_logs_from_queue)

    def train_model(self):
        if self.training_in_progress:
            return
        try:
            train_val_ratio = float(self.split_var.get())
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            lr = float(self.lr_var.get())
            wd = float(self.wd_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid hyperparameter values.")
            return

        train_loader, val_loader = build_dataloaders(train_val_ratio, batch_size)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        self.training_in_progress = True
        self.train_button.config(state=DISABLED)
        self.classify_button.config(state=DISABLED)
        self.realtime_button.config(state=DISABLED)
        self.eval_button.config(state=DISABLED)
        self.log_text.delete("1.0", tk.END)
        self.progress_var.set(0.0)

        def re_enable_buttons():
            self.train_button.config(state=NORMAL)
            self.classify_button.config(state=NORMAL)
            self.realtime_button.config(state=NORMAL)
            self.eval_button.config(state=NORMAL)
            self.training_in_progress = False

        t = threading.Thread(target=train_thread_fn, args=(
            self.classifier,
            epochs,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            self.log_message,
            self.progress_var,
            re_enable_buttons
        ), daemon=True)
        t.start()

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pth",
                                                  filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if file_path:
            torch.save(self.classifier.state_dict(), file_path)
            messagebox.showinfo("Saved", f"Model saved to {file_path}") 

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if not file_path:
            return
        try:
            model = build_model(num_classes)
            freeze_clip_layers(model, unfreeze_last=4)
            state_dict = torch.load(file_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            self.classifier = model

            messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{file_path}")
            self.classify_button.config(state=NORMAL)
            self.realtime_button.config(state=NORMAL)
            self.eval_button.config(state=NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}")

    def classify_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All Files", "*.*")])
        if not file_path:
            return

        img = Image.open(file_path).convert("RGB")
        display_img = img.resize((200, 200), resample=Image.Resampling.LANCZOS)
        self.current_image_tk = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=self.current_image_tk)

        img_tensor = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = nn.Softmax(dim=1)(outputs).squeeze().cpu().numpy()

        self.prob_text.delete("1.0", tk.END)
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            label = emotion_classes[idx]
            prob = probs[idx] * 100
            self.prob_text.insert(tk.END, f"{label}: {prob:.2f}%\n")

    def realtime_detection(self):
        real_time_detection(self.classifier)

    def show_confusion_matrix(self):
        try:
            _, val_loader = build_dataloaders(train_val_ratio=0.8, batch_size=32)
            compute_confusion_matrix_report(self.classifier, val_loader, self.log_message, mode="Trained Model")
        except Exception as e:
            messagebox.showerror("Error", f"Could not evaluate: {e}")

    def show_zero_shot_eval(self):
        try:
            _, val_loader = build_dataloaders(train_val_ratio=0.8, batch_size=32)
            acc, preds, labels = zero_shot_eval(val_loader)
            self.log_message(f"Zero-Shot Accuracy: {acc:.2f}%")
            compute_confusion_matrix_report((preds, labels), val_loader, self.log_message, mode="Zero-Shot")
        except Exception as e:
            messagebox.showerror("Error", f"Could not run zero-shot evaluation: {e}")
