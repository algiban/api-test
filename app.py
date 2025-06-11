from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("model/model_buah.h5")

# Label sesuai model
labels = ['alpukat', 'anggur', 'apel', 'belimbing', 'blueberry', 'buah naga', 'ceri', 'delima', 'duku', 'durian', 'jambu air', 'jambu biji', 'jeruk', 'kelapa', 'kiwi', 'kurma', 'leci', 'mangga', 'manggis', 'markisa', 'melon', 'nanas', 'nangka', 'pepaya', 'pir', 'pisang', 'rambutan', 'salak', 'sawo', 'semangka', 'sirsak', 'stroberi', 'tomat']

# Path file JSON
JSON_FILE = 'predictions.json'

# Fungsi menyimpan hasil prediksi ke file JSON
def save_prediction_to_json(label, confidence):
    data = {
        "label": label,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.insert(0, data)  # Tambahkan data terbaru di awal

    with open(JSON_FILE, 'w') as f:
        json.dump(existing_data, f, indent=4)

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
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_label = labels[class_index]

        # Simpan ke file JSON
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
        return jsonify([])  # Jika file tidak ada, kembalikan list kosong

    with open(JSON_FILE, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []

    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
