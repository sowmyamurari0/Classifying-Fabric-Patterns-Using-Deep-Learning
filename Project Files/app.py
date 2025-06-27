from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model once when the app starts
model = tf.keras.models.load_model('fabric_model_final.h5')

# Your class names (change if needed)
class_names = ['checked', 'dot', 'floral', 'snake_skin', 'stripe', 'zigzag']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    predicted = request.args.get('predicted')
    label = request.args.get('label')
    return render_template('home.html', predicted=predicted, label=label)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        file.save(filepath)

        # Preprocess the image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))  # adjust if your model uses different size
        img = img / 255.0  # normalize if needed
        img = np.expand_dims(img, axis=0)

        # Predict
        predictions = model.predict(img)
        predicted_label = class_names[np.argmax(predictions)]

        return redirect(url_for('home', predicted=1, label=predicted_label))

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
