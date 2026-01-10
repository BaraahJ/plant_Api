from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

app = FastAPI(title="Plant Identifier API")
@app.get("/")
def health_check():
    return {"status": "ok"}

# --- مسارات الموديل والليبلز ---
MODEL_PATH = "model/plant_model_quant.tflite"
LABELS_PATH = "model/labels.txt"

# --- تحميل الليبلز ---
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- تحميل الموديل TFLite ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'][1:3]

# --- دالة تجهيز الصورة ---
def preprocess_image(image: Image.Image):
    image = image.resize(INPUT_SHAPE)
    image_array = np.array(image).astype(np.float32)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array]*3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- دالة التنبؤ ---
def predict_plant(image_array: np.ndarray):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])
    confidence = float(np.max(output_data[0]))
    return predicted_class, confidence

# --- API endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        input_data = preprocess_image(image)
        predicted_class, confidence = predict_plant(input_data)
        return {
            "plant_name": class_names[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
