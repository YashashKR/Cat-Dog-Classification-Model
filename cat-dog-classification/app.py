from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model (ensure the path to your model is correct)
model = load_model(r'C:\Users\Yashuyashash\Downloads\final_cat_dog_classifier.keras')

# Define the categories (assuming binary classification: cat vs dog)
categories = ['Cat', 'Dog']

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading and predicting the image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        
        if file:
            # Save the uploaded image
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            
            # Preprocess the image for prediction
            img = image.load_img(file_path, target_size=(150, 150))  # Change target_size to (150, 150)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale the image
            
            # Make the prediction
            prediction = model.predict(img_array)
            
            # Determine the class
            predicted_class = categories[int(prediction > 0.5)]  # Binary classification: >0.5 is dog, else cat
            
            return render_template('index.html', filename=file.filename, prediction=predicted_class)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
