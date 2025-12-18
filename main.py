from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

app = FastAPI(title="Plant Identifier API")

# --- إعداد الموديل TFLite ---
MODEL_PATH = "model/plant_model_quant.tflite"  # عدل المسار إذا لزم
LABELS_PATH = "model/labels.txt"               # ملف labels من Colab

# تحميل الموديل
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# معلومات المدخلات والمخرجات
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SHAPE = input_details[0]['shape'][1:3]  # [height, width]

# تحميل الـ labels
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
with open(LABELS_PATH, 'r') as f:
    class_names = [line.strip() for line in f]

# --- دالة تحضير الصورة ---
def preprocess_image(image: Image.Image):
    # تحويل للصيغة RGB للتأكد
    image = image.convert('RGB')
    # تغيير الحجم
    image = image.resize(INPUT_SHAPE)
    # تحويل إلى numpy array وتطبيع
    image_array = np.array(image).astype(np.float32) / 255.0
    # إضافة بعد batch
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- دالة التنبؤ ---
def predict_plant(image_array: np.ndarray):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))
    return predicted_index, confidence

# --- API endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        input_data = preprocess_image(image)
        predicted_index, confidence = predict_plant(input_data)
        return {
            "plant_name": class_names[predicted_index],
            "confidence": confidence
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
