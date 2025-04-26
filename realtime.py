import cv2
import torch
import time
from PIL import Image
from data import data_transform, emotion_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def real_time_detection(classifier):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    UPDATE_INTERVAL = 3.0
    last_update_time = time.time()
    stored_face_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        now = time.time()
        if (now - last_update_time) >= UPDATE_INTERVAL:
            stored_face_results = []
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                img_tensor = data_transform(img_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = classifier(img_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                sorted_indices = probs.argsort()[::-1]
                emotions_list = [(emotion_classes[idx], probs[idx]) for idx in sorted_indices]
                stored_face_results.append(((x, y, w, h), emotions_list))
            last_update_time = now

        for (box, emotion_data) in stored_face_results:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for i, (emotion_label, prob) in enumerate(emotion_data):
                label_str = f"{emotion_label}: {prob*100:.1f}%"
                cv2.putText(frame, label_str, (x, y - 10 - i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
