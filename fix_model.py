import json
import os
import zipfile

SRC = "breast_mammography_model.keras"
DST = "breast_mammography_model_fixed.keras"


def show_layers(cfg):
    layers = cfg.get("config", {}).get("layers", [])
    print("Top level layers:")
    for lyr in layers:
        print("  -", lyr.get("class_name"), "|", lyr.get("config", {}).get("name"))
        inner = lyr.get("config", {})
        if "output_layers" in inner:
            print("      output_layers:", inner["output_layers"])


def dedupe_outputs(node):
    """If a nested model lists the same output layer twice, keep only one."""
    changed = 0
    if isinstance(node, dict):
        ol = node.get("output_layers")
        if isinstance(ol, list) and len(ol) > 1 and all(isinstance(x, list) for x in ol):
            if len({x[0] for x in ol}) == 1:
                node["output_layers"] = [ol[0]]
                changed += 1
        for v in node.values():
            changed += dedupe_outputs(v)
    elif isinstance(node, list):
        for v in node:
            changed += dedupe_outputs(v)
    return changed


with zipfile.ZipFile(SRC) as zin:
    print("Files inside:", zin.namelist())
    cfg = json.loads(zin.read("config.json"))
    show_layers(cfg)
    n = dedupe_outputs(cfg)
    print("Fixed duplicate outputs:", n)

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(cfg).encode("utf-8")
            zout.writestr(item, data)

print("Saved:", DST, round(os.path.getsize(DST) / 1024 / 1024, 1), "MB")

import tensorflow as tf

try:
    m = tf.keras.models.load_model(DST, compile=False)
    print("LOAD OK", m.input_shape, m.output_shape)
except Exception as e:
    print("LOAD FAILED:", e)
