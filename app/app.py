import os
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import tensorflow as tf

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads/'
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('hierarchical_model.keras')

# Load the tokenizer and image features
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('features.pkl', 'rb') as file:
    img_features = pickle.load(file)

# Define max_caption_length (Ensure it's the same as the one used during training)
max_caption_length = 35

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def get_word_from_index(index, tokenizer):
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

def generate_caption(image_feature, model, tokenizer, max_caption_length):
    caption = 'startseq'
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_feature, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        if predicted_word is None or predicted_word == 'endseq':
            break
        caption += " " + predicted_word
    return caption.replace('startseq', '').replace('endseq', '')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        image = preprocess_image(filepath)
        print(filepath)
        # Extract image ID
        image_id = file.filename.split('.')[0]
        image_feature = img_features.get(image_id)

        if image_feature is None:
            flash("Image not found in dataset. Please use a valid image.")
            return redirect(request.url)

        # Generate caption
        caption = generate_caption(image_feature, model, tokenizer, max_caption_length)
        
        return render_template('result.html', image_url=filepath, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
