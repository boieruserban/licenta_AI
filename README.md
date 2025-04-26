Proiect de Licență - Detecția Emoțiilor cu CLIP
Acest proiect implementează un sistem de detecție a emoțiilor în imagini statice și în timp real, folosind modelul CLIP de la OpenAI, adaptat pentru clasificarea expresiilor faciale.

**Python Version:** 3.9.13
Structura proiectului

licenta_AI/
├── data.py          # Transformări de date și încărcare dataset
├── gui.py           # Interfață grafică cu Tkinter
├── main.py          # Script principal pentru rularea aplicației
├── model.py         # Definirea și configurarea modelului CLIPClassifier
├── realtime.py      # Detecție emoții în timp real prin webcam
├── train.py         # Funcții de antrenare și validare model
├── requirements.txt # Lista dependențelor necesare
├── README.md        # Documentația proiectului
├── .gitignore       # Fișiere ignorate de Git
├── venv/            # Mediu virtual (local, nu urcat pe GitHub)

Librării folosite

- **PyTorch (`torch`)** – Framework principal pentru antrenarea și rularea modelului CLIP.
- **Torchvision (`torchvision`)** – Transformări și procesare imagini necesare antrenării modelelor.
- **Transformers (`transformers`)** – Încărcarea și utilizarea modelului CLIP de la HuggingFace.
- **Datasets (`datasets`)** – Accesarea și manipularea datasetului `human_face_emotions_roboflow`.
- **OpenCV (`opencv-python`)** – Accesarea webcam-ului și prelucrarea imaginilor video în timp real.
- **Pillow (`pillow`)** – Procesare și conversie imagini între formate necesare pentru Tkinter și OpenCV.
- **NumPy (`numpy`)** – Operații numerice eficiente pentru procesarea datelor de imagine.
- **Tkinter (`tk`)** – Crearea interfeței grafice GUI pentru utilizator.


Cum rulezi proiectul

1. Clonează repository-ul sau descarcă-l local.

2. Creează și activează mediul virtual:
```bash
    python -m venv venv
    venv\Scripts\activate
```
3. Instalează dependențele necesare:
```bash
    pip install -r requirements.txt
```
4. Rulează aplicația:
```bash
    python main.py
```

Modelul Antrenat (.pth)
Modelul antrenat nu este inclus în repository din cauza limitărilor de dimensiune GitHub.
Poate fi descărcat separat de la următorul link:

https://drive.google.com/file/d/1Qd0rwC4tKcMVrpw5EnmJCMc8lhRqE1SW/view?usp=sharing

Modelul `.pth` trebuie plasat în același folder cu fișierele sursă (`main.py`, `model.py`, etc.).
Funcționalități implementate

-  Antrenarea modelului CLIP pentru clasificarea emoțiilor
-  Testarea modelului pe imagini statice
-  Accesarea camerei video folosind OpenCV
-  Detecția emoțiilor în timp real prin webcam
-  Salvarea și încărcarea modelelor antrenate (`.pth`)
-  Interfață grafică simplă și intuitivă cu Tkinter
-  Bară de progres și jurnalizare a procesului de antrenare

Dataset utilizat
• [tukey/human_face_emotions_roboflow](https://huggingface.co/datasets/tukey/human_face_emotions_roboflow)

Dataset public pentru clasificarea expresiilor faciale în emoții de bază.



