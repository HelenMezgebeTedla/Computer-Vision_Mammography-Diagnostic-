import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 150
CLASS_NAMES = ['Benign', 'Malignant']

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not open processing image path asset target at: {image_path}")
    resized = cv2.resize(img, (150, 150), interpolation=cv2.INTER_LINEAR)
    img_array = np.array(resized, dtype=np.float32)
    processed_tensor = preprocess_input(img_array)
    return processed_tensor
