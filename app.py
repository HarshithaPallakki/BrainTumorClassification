from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

# Load the trained Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Load the model weights or pickle file here
import pickle

# Load the trained Random Forest Classifier model
with open("model.pkl", "rb") as file:
    rf_classifier = pickle.load(file)


# Define the classes
classes = {0: "No Tumour", 1: "Pituitary Tumour", 2: "Meningioma Tumour", 3: "Glioma Tumour"}

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (200, 200))
        image_flattened = np.array(image).flatten()
        
        # Perform the prediction using the trained Random Forest Classifier
        predicted_class = rf_classifier.predict([image_flattened])
        predicted_label = classes[predicted_class[0]]
        
        # Render the result page with the predicted class label
        return render_template('result.html', predicted_class=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
