Proiect de Licență - Detecția Emoțiilor cu CLIP
Acest proiect implementează un sistem de detecție a emoțiilor în imagini statice și în timp real, folosind modelul CLIP de la OpenAI, adaptat pentru clasificarea expresiilor faciale.

**Python Version:** 3.9.13
Structura proiectului

Structura proiectului
licenta_AI/
├── data.py                   # Preprocesare și încărcare dataset (HuggingFace Datasets)
├── evaluate.py               # Generare matrice de confuzie și rapoarte de clasificare
├── gui.py                    # Interfață grafică (Tkinter) pentru antrenare/testare model
├── main.py                   # Punct de intrare aplicație (lansare GUI)
├── model.py                  # Definirea arhitecturii CLIPClassifier
├── realtime.py               # Detecție emoții în timp real cu OpenCV și DNN face detector
├── train.py                  # Funcții de antrenare + evaluare (inclusiv Zero-Shot CLIP)
├── deploy.prototxt           # Configurație rețea pentru face detector (Caffe model)
├── res10_300x300_ssd_iter_140000.caffemodel # Model binar pentru face detector
├── requirements.txt          # Lista dependențelor
├── README.md                 # Documentația proiectului
├── .gitignore                # Fișiere ignorate (venv, pyc, cache etc.)
└── venv/                     # Mediu virtual (nu se urcă pe GitHub)

⚙️ Librării folosite

PyTorch (torch) – antrenare și inferență model.

Torchvision – transformări de imagini.

Transformers (HuggingFace) – încărcarea modelului CLIP.

Datasets – încărcarea dataset-ului tukey/human_face_emotions_roboflow.

OpenCV (cv2) – detecția fețelor + accesarea webcam-ului.

Pillow (PIL) – procesare imagini.

NumPy – operații numerice.

Scikit-learn – clasificare și matrice de confuzie.

Matplotlib & Seaborn – vizualizare rezultate.

Tkinter – interfața grafică.

 Cum rulezi proiectul

Clonează repository-ul sau descarcă local:

git clone <repo_url>
cd licenta_AI


Creează și activează mediul virtual:

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate


Instalează dependențele:

pip install -r requirements.txt


Rulează aplicația GUI:

python main.py

Funcționalități

Antrenare model CLIPClassifier pe dataset-ul de emoții.

Evaluare clasică: matrice de confuzie + rapoarte de clasificare.

Evaluare Zero-Shot cu CLIP (fără fine-tuning).

Clasificare imagini statice prin GUI.

Detecție emoții în timp real folosind webcam + face detector DNN.

Interfață grafică intuitivă (Tkinter) cu:

setarea hiperparametrilor (batch size, epochs, learning rate, etc.)

progres bar pentru antrenare

log-uri detaliate în timp real

butoane pentru train, save, load, evaluate

Modelul antrenat

Modelul .pth nu este inclus în repository din cauza limitărilor GitHub.
Poate fi descărcat separat de la:

https://drive.google.com/file/d/1Qd0rwC4tKcMVrpw5EnmJCMc8lhRqE1SW/view

Plasează fișierul descărcat în directorul principal al proiectului.

Dataset

https://huggingface.co/datasets/tukey/human_face_emotions_roboflow
→ conține imagini etichetate pe clase de emoții de bază (fericire, tristețe, furie, surpriză etc.).



