from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from model import load_model, predict_image
from utils import preprocess_image


app = Flask(__name__, template_folder='../frontend/templates')


# Load the trained DenseNet model
model = load_model()

# Define the folder for uploading images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Preprocess image and make prediction
    image = preprocess_image(file_path, size=256)
    prediction = predict_image(model, image)

    os.remove(file_path)  # Optionally delete the uploaded image after processing

    # Return result
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
