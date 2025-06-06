from __future__ import annotations
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from data import data_transform, emotion_classes          

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model files (relative paths are resolved next to this .py file)
_THIS_DIR = Path(__file__).resolve().parent
FACE_PROTO  = str(_THIS_DIR / "deploy.prototxt")
FACE_MODEL = str(_THIS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")

# Load the face detector
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Use CUDA for the detector if OpenCV is built with it and the user has a GPU
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
else:
    # Default to CPU – this works everywhere
    face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ---------------------------------------------------------------
# Face-detection helper
# ---------------------------------------------------------------
def detect_faces_dnn(
    frame: np.ndarray,
    net: cv2.dnn_Net,
    conf_threshold: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    """
    Runs the OpenCV DNN SSD face detector on a single BGR frame and
    returns a list of bounding boxes (x1, y1, x2, y2).
    """
    (h, w) = frame.shape[:2]

    # Prepare blob & forward
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()              # shape = (1, 1, N, 7)

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        # Extract box
        (x1, y1, x2, y2) = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if (x2 - x1) > 10 and (y2 - y1) > 10:   # ignore tiny boxes
            boxes.append((x1, y1, x2, y2))

    return boxes


# ---------------------------------------------------------------
# Real-time loop
# ---------------------------------------------------------------
def real_time_detection(classifier: torch.nn.Module) -> None:
    """
    Opens the default webcam, finds faces with OpenCV DNN, predicts
    emotions with the supplied classifier, and overlays results.
    Press 'q' to quit.
    """
    # Put classifier in eval mode & move to correct device once
    classifier.to(device).eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    UPDATE_INTERVAL_SEC = 2.0          # how often to re-evaluate faces
    last_update_time = 0.0
    cached_results: List[Tuple[Tuple[int, int, int, int], List[Tuple[str, float]]]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Empty frame – terminating.")
                break

            now = time.time()

            # Re-run the detector & classifier at the desired interval
            if (now - last_update_time) >= UPDATE_INTERVAL_SEC:
                last_update_time = now
                cached_results.clear()

                boxes = detect_faces_dnn(frame, face_net, conf_threshold=0.6)

                for (x1, y1, x2, y2) in boxes:
                    face_bgr = frame[y1:y2, x1:x2]
                    if face_bgr.size == 0:
                        continue

                    # Convert to PIL RGB for transform pipeline
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    pil_img  = Image.fromarray(face_rgb)
                    tensor   = data_transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = classifier(tensor)
                        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    # Sort descending
                    top_indices = probs.argsort()[::-1]
                    emotions = [(emotion_classes[i], float(probs[i])) for i in top_indices]
                    cached_results.append(((x1, y1, x2, y2), emotions))

            # ------------------------------------------------------------------
            # Draw cached results (we always draw, even when not updating,
            # so the overlay stays visible while we wait for the next update)
            # ------------------------------------------------------------------
            for ((x1, y1, x2, y2), emotions) in cached_results:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Show the top-3 emotions
                for idx, (label, prob) in enumerate(emotions):
                    text = f"{label}: {prob * 100:.1f}%"
                    y_text = y1 - 10 - idx * 20
                    if y_text < 10:         # keep text on-screen
                        y_text = y1 + 20 + idx * 20
                    cv2.putText(
                        frame,
                        text,
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("Real-Time Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):        # q or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()