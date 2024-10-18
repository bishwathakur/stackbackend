from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
import cv2
from PIL import Image

# Load the Keras model
model = load_model('model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Hello World</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file provided", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Ensure the uploaded file is a JPG image
        if not file.filename.lower().endswith('.jpg'):
            return "Unsupported file type. Please upload a JPG image.", 400
        
        # Convert the uploaded file to a PIL image
        imag = Image.open(file)
        img = np.array(imag)

        plt.figure(figsize=(6,4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        img = cv2.resize(img, (224, 224))
        img = np.reshape(img, [-1, 224, 224,3])
        result = np.argmax(model.predict(img))
        # result = model.predict(img)
        if result == 0: ans="This image is Recyclable"
        elif result ==1: ans="This image is Organic"

        # Return the result as a plain string
        return f"Predicted Class: {ans}"

    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
