import json
import os
import tempfile
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

SRC = "breast_mammography_model.keras"
DST = "breast_mammography_model_rebuilt.keras"

# 1. Read the settings and weights from the old file
with zipfile.ZipFile(SRC) as z:
    cfg = json.loads(z.read("config.json"))
    tmp = tempfile.mkdtemp()
    z.extract("model.weights.h5", tmp)
    weights_path = os.path.join(tmp, "model.weights.h5")

saved = cfg["config"]["layers"]
dense_cfgs = [l["config"] for l in saved if l["class_name"] == "Dense"]
drop_cfgs = [l["config"] for l in saved if l["class_name"] == "Dropout"]

for c in dense_cfgs + drop_cfgs:
    print("Saved layer settings:", c.get("name"), c.get("units"), c.get("activation"), c.get("rate"))

d0, d1 = dense_cfgs[0], dense_cfgs[1]
drop = drop_cfgs[0]

# 2. Build the same model in code
base = ResNet50(include_top=False, weights=None, input_shape=(150, 150, 3), name="resnet50")

model = models.Sequential(
    [
        layers.Input(shape=(150, 150, 3), name="input_layer_1"),
        base,
        layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
        layers.Dense(d0["units"], activation=d0["activation"], name=d0["name"]),
        layers.Dropout(drop["rate"], name=drop["name"]),
        layers.Dense(d1["units"], activation=d1["activation"], name=d1["name"]),
    ]
)

# 3. Load the trained weights
model.load_weights(weights_path)
print("Weights loaded OK")
print("Input shape:", model.input_shape, "| Output shape:", model.output_shape)

# 4. Quick test
test = np.random.rand(1, 150, 150, 3).astype("float32")
print("Test prediction:", model.predict(test, verbose=0))

# 5. Save a clean new model file
model.save(DST)
print("Saved:", DST, round(os.path.getsize(DST) / 1024 / 1024, 1), "MB")

# 6. Check that the new file loads
m2 = tf.keras.models.load_model(DST, compile=False)
print("RELOAD OK", m2.output_shape)
