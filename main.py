import tkinter as tk
import torch
from model import build_model, freeze_clip_layers
from data import num_classes
from gui import GUI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = build_model(num_classes).to(device)
freeze_clip_layers(classifier, unfreeze_last=4)

root = tk.Tk()
app = GUI(root, classifier)
root.mainloop()
