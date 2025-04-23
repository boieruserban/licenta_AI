# Detecția Emoțiilor cu CLIP - Proiect de Licență

Acest proiect implementează un sistem de detecție a emoțiilor în imagini și în timp real, folosind modelul CLIP de la OpenAI, fine-tuned pe un set de date de expresii faciale.

---

##  Tehnologii folosite

- [PyTorch](https://pytorch.org/) – pentru definirea și antrenarea modelului
- [Transformers](https://huggingface.co/transformers/) – pentru CLIPModel de la OpenAI
- [Tkinter](https://docs.python.org/3/library/tkinter.html) – interfață grafică
- [OpenCV](https://opencv.org/) – captură și procesare video
- [HuggingFace Datasets](https://huggingface.co/datasets) – pentru încărcarea datasetului de antrenament

---

##  Structura proiectului

```bash
licenta_AI/
├── features/
│   ├── data.py               # Încărcare dataset + transformări
│   ├── gui.py                # Interfața grafică
│   ├── model.py              # Arhitectura modelului CLIPClassifier
│   ├── realtime.py           # Webcam + detecție în timp real
│   └── train.py              # Funcții de antrenare și validare
├── variantatest.pth                # Model de test
├── trainedmodelnew.pth             # Model salvat(best val loss)
├── main.py                         # Script principal pentru rulare GUI
├── requirements.txt                # Dependențe Python
└── README.md                       # Acest fișier
