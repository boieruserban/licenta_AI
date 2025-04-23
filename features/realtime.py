import time
import cv2
import torch
import numpy as np
from PIL import Image
from tkinter import filedialog, Toplevel
import tkinter as tk

def real_time_detection(classifier, data_transform, emotion_classes, device):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    recording = False
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = ""

    UPDATE_INTERVAL = 3.0  # seconds
    last_update_time = time.time()
    stored_face_results = []

    print("📷 Camera started. Press 'r' to record, 's' to stop, 'p' for snapshot, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        now = time.time()
        should_update = (now - last_update_time) >= UPDATE_INTERVAL

        if should_update:
            stored_face_results = []
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                img_tensor = data_transform(img_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = classifier(img_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                sorted_indices = np.argsort(probs)[::-1]
                emotions_list = [(emotion_classes[idx], probs[idx]) for idx in sorted_indices]
                stored_face_results.append(((x, y, w, h), emotions_list))
            last_update_time = now

        for (box, emotion_data) in stored_face_results:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            for i, (emotion_label, prob) in enumerate(emotion_data):
                label_str = f"{emotion_label}: {prob*100:.1f}%"
                cv2.putText(frame, label_str, (x, y - 10 - i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if recording and video_writer:
            video_writer.write(frame)

        cv2.imshow("Real-Time Emotion Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            print("🔴 Recording started. Press 's' to stop.")
            sub_root = Toplevel()
            sub_root.withdraw()
            output_path = filedialog.asksaveasfilename(
                defaultextension=".avi",
                filetypes=[("AVI files", "*.avi"), ("All Files", "*.*")],
                title="Save the video file as..."
            )
            sub_root.destroy()
            if output_path:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                recording = True

        elif key == ord('s') and recording:
            print(f"🛑 Recording stopped. Video saved at: {output_path}")
            recording = False
            if video_writer:
                video_writer.release()
                video_writer = None

        elif key == ord('p'):
            sub_root = Toplevel()
            sub_root.withdraw()
            snapshot_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All Files", "*.*")],
                title="Save Snapshot"
            )
            sub_root.destroy()
            if snapshot_path:
                cv2.imwrite(snapshot_path, frame)
                print(f"📸 Snapshot saved to {snapshot_path}")

        elif key == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
