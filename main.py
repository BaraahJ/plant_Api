from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from typing import List

app = FastAPI(title="Plant Identifier API")

# --------- إعداد الموديل ---------
MODEL_PATH = "model/plant_model.tflite"    # استخدم الموديل الكبير غير quantized
LABELS_PATH = "model/labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'][1:3]   # [H, W]

# تحميل labels
with open(LABELS_PATH, 'r') as f:
    class_names = [line.strip() for line in f]


# --------- تحضير الصورة ---------
def preprocess_image(image: Image.Image):
    image = image.convert('RGB')

    # رفع الجودة → anti-aliasing
    image = image.resize(INPUT_SHAPE, Image.Resampling.LANCZOS)

    image_array = np.array(image).astype(np.float32)

    # NORMALIZATION المناسب للموديلات الحديثة
    image_array = image_array / 255.0  

    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# --------- التنبؤ ---------
def predict_plant(image_array: np.ndarray):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])[0]

    # softmax manually
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    predicted_index = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return predicted_index, confidence, probs


# --------- API للصورة الواحدة ---------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        img_array = preprocess_image(image)

        idx, conf, probs = predict_plant(img_array)
        return {
            "plant_name": class_names[idx],
            "confidence": conf,
            "probabilities": {class_names[i]: float(p) for i, p in enumerate(probs)}
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# --------- API لثلاث صور مع التصويت Voting ---------
@app.post("/predict-multi")
async def predict_multi(files: List[UploadFile] = File(...)):
    if len(files) != 3:
        return JSONResponse(
            status_code=400,
            content={"error": "Send exactly 3 images."}
        )

    results = []
    votes = {}

    for f in files:
        image = Image.open(io.BytesIO(await f.read()))
        img_array = preprocess_image(image)
        idx, conf, _ = predict_plant(img_array)

        plant = class_names[idx]
        results.append({"plant": plant, "confidence": conf})

        votes[plant] = votes.get(plant, 0) + 1

    # اختيار الفئة بالأغلبية
    final_plant = max(votes, key=votes.get)

    return {
        "images_results": results,
        "final_prediction": final_plant,
        "votes": votes
    }
