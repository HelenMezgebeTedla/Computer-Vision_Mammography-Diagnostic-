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

## Model

Transfer learning with **ResNet50**: Input (150 x 150 x 3) > ResNet50 > GlobalAveragePooling2D > Dense(128, ReLU) > Dropout(0.4) > Dense(1, Sigmoid).

- Score of 0.5 or higher: **Malignant**
- Score below 0.5: **Benign**

Images are converted to RGB, resized to 150 x 150, and passed through ResNet50 `preprocess_input`.

## Results

| Metric | Value |
| --- | --- |
| Accuracy | [FILL IN] |
| Recall | [FILL IN] |
| AUC | [FILL IN] |

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

## Known issue: Keras version

The model was saved with Keras 3 and failed to load on Keras 2.15. After upgrading to TensorFlow 2.20 it still failed (`Layer "dense" expects 1 input(s), but it received 2`). `rebuild_model.py` fixes this by building the same model in code and loading the saved weights. No retraining is needed. Use TensorFlow 2.16 or higher.

## Limitations

Small 150 x 150 images lose fine detail. The model was trained on limited data and has no clinical testing. The confidence score is not a medical probability.

**Author:** Helen Mezgebe Tedla
