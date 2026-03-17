from tensorflow.keras.models import load_model as tf_load_model
import numpy as np

MODEL_PATH = 'D://DDLP//MURA_model.h5'

def load_model():
    model = tf_load_model(MODEL_PATH)
    return model

def predict_image(model, image):
    # Add batch dimension to the image
    image = np.expand_dims(image, axis=0)
    
    # Predict the label (binary classification: positive or negative)
    prediction = model.predict(image)
    return 'positive' if prediction[0] > 0.5 else 'negative'
