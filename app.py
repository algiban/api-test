from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS
import json
from datetime import datetime
import os
import gdown

app = Flask(__name__)
CORS(app)

# --- Setup Model ---
MODEL_PATH = "model/model_buah.h5"
GOOGLE_DRIVE_ID = "1ZfRZXMj4qiBSRKookrB3SW5w_IWFrv-S"
MODEL_DIR = "model"

# Cek apakah model belum ada → download sekali
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model setelah terdownload
model = tf.keras.models.load_model(MODEL_PATH)

# Label hasil training (urutan harus sesuai dengan model)
labels = ['alpukat', 'anggur', 'apel', 'belimbing', 'blueberry', 'buah naga', 'ceri', 'delima', 'duku', 'durian',
          'jambu air', 'jambu biji', 'jeruk', 'kelapa', 'kiwi', 'kurma', 'leci', 'mangga', 'manggis', 'markisa',
          'melon', 'nanas', 'nangka', 'pepaya', 'pir', 'pisang', 'rambutan', 'salak', 'sawo', 'semangka',
          'sirsak', 'stroberi', 'tomat']

# File penyimpanan hasil prediksi
JSON_FILE = 'predictions.json'

# Fungsi simpan hasil prediksi ke file
def save_prediction_to_json(label, confidence):
    data = {
        "label": label,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
    else:
        existing_data = []

    existing_data.insert(0, data)

    with open(JSON_FILE, 'w') as f:
        json.dump(existing_data, f, indent=4)

# --- Routes ---
@app.route('/')
def index():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img = Image.open(file.stream).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        predicted_label = labels[class_index]

        save_prediction_to_json(predicted_label, confidence)

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    if not os.path.exists(JSON_FILE):
        return jsonify([])

    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []

    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
