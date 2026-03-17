import cv2
import numpy as np

def preprocess_image(image_path, size=256):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize pixel values
    return image
