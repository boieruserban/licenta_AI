Proiect de Licență - Detecția Emoțiilor cu CLIP
Acest proiect implementează un sistem de detecție a emoțiilor în imagini statice și în timp real, folosind modelul CLIP de la OpenAI, adaptat pentru clasificarea expresiilor faciale.

**Python Version:** 3.9.13

##  Structura proiectului

    licenta_AI/
    ├── data.py                   # Preprocesare și încărcare dataset
    ├── evaluate.py               # Matrice de confuzie & rapoarte de clasificare
    ├── gui.py                    # Interfață grafică (Tkinter)
    ├── main.py                   # Punct de intrare aplicație (GUI)
    ├── model.py                  # Arhitectura CLIPClassifier
    ├── realtime.py               # Detecție emoții în timp real (OpenCV + DNN face detector)
    ├── train.py                  # Antrenare + evaluare + Zero-Shot CLIP
    ├── deploy.prototxt           # Configurație rețea detector fețe (Caffe)
    ├── res10_300x300_ssd_iter_140000.caffemodel   # Modelul detectorului de fețe
    ├── requirements.txt          # Dependențe
    ├── README.md                 # Documentație
    └── .gitignore                # Fișiere ignorate

---

##  Tehnologii & Librării
- **PyTorch** – rețele neuronale
- **Torchvision** – transformări de imagini
- **Transformers (HuggingFace)** – modelul CLIP
- **Datasets** – încărcarea dataset-ului `tukey/human_face_emotions_roboflow`
- **OpenCV** – acces webcam + detector fețe (SSD)
- **Pillow** – procesare imagini
- **Scikit-learn** – rapoarte de clasificare / matrice de confuzie
- **Matplotlib & Seaborn** – vizualizări
- **Tkinter** – interfață grafică

---

##  Instalare & Rulare

1) Clonează repository-ul

        git clone <repo_url>
        cd licenta_AI

2) Creează și activează mediul virtual

        python -m venv venv

   Windows:

        venv\Scripts\activate

   Linux / Mac:

        source venv/bin/activate

3) Instalează dependențele

        pip install -r requirements.txt

4) Rulează aplicația GUI

        python main.py

---

##  Funcționalități
- Antrenarea modelului **CLIPClassifier** pe dataset-ul de emoții
- Evaluare clasică: matrice de confuzie & rapoarte de clasificare
- Evaluare **Zero-Shot CLIP** (fără fine-tuning)
- Clasificare imagini statice din GUI
- Detecție emoții în timp real (webcam) cu **SSD face detector**
- Interfață grafică cu:
  - setare hiperparametri (epochs, batch size, learning rate, weight decay)
  - bară de progres pentru antrenare
  - log-uri în timp real
  - butoane: **Train / Save / Load / Evaluate / Zero-Shot / Real-time**

---

##  Modelul antrenat
Modelul `.pth` nu este inclus în repository (limitări GitHub). Poate fi descărcat de aici:

🔗 Google Drive: https://drive.google.com/file/d/1Qd0rwC4tKcMVrpw5EnmJCMc8lhRqE1SW/view?usp=sharing

Plasează fișierul descărcat în directorul principal al proiectului.

---

##  Dataset
**tukey/human_face_emotions_roboflow** – imagini etichetate pe emoții de bază (fericire, tristețe, furie, surpriză etc.).  
HuggingFace: https://huggingface.co/datasets/tukey/human_face_emotions_roboflow



