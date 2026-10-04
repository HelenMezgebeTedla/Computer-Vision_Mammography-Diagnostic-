# Mammography Diagnostic Portal

A deep learning project that classifies mammography images as **Benign** or **Malignant**, with a Streamlit web app to test it.

> **Academic notice:** This is a coursework prototype for an AkiraChix DAS assignment. It is NOT a medical tool, is not clinically validated, and must never be used on real patients.

## What is inside

| File | Purpose |
| --- | --- |
| `Computer Vision.ipynb` | Data loading, training, and evaluation |
| `preprocess.py` | Image preprocessing |
| `app.py` | Streamlit app: upload an image, get a prediction |
| `rebuild_model.py` | Rebuilds the model and loads the trained weights (fixes a Keras version error) |
| `fix_model.py` | Early repair attempt, kept for reference |
| `csv/` | Labels and metadata |
| `requirements.txt` | Python packages |


## Run it

```bash
git clone https://github.com/HelenMezgebeTedla/Computer-Vision_Mammography-Diagnostic-.git
cd Computer-Vision_Mammography-Diagnostic-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put your trained model (`breast_mammography_model.keras`) in the folder, then:

```bash
python3 rebuild_model.py
streamlit run app.py
```

Open `http://localhost:8501`.

If the upload shows a 403 error, run:

```bash
streamlit run app.py --server.enableXsrfProtection=false --server.enableCORS=false
```

## Limitations

Small 150 x 150 images lose fine detail. The model was trained on limited data and has no clinical testing. The confidence score is not a medical probability.

**Author:** Helen Mezgebe Tedla
