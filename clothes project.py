from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from flask import jsonify  

app = Flask(__name__)
app.config['DATASET_FOLDER'] = 'static'  
app.config['UPLOAD_FOLDER'] = 'uploads/'  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to extract image features
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  
    features = image.flatten()  
    return features

# Load images and extract features
dataset_path = './static'  
images = []
features_list = []

for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(dataset_path, filename)
        features = extract_features(img_path)
        features_list.append(features)
        images.append(filename)

print(f"Number of images loaded: {len(images)}")  # Check number of images

# Normalize features
scaler = StandardScaler()
features_list_normalized = scaler.fit_transform(features_list)

# Create KNN model
n_neighbors = min(4, len(features_list))  # Adjust number of neighbors
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(features_list_normalized)

# Function to find similar images
def find_similar_images(input_image_path):
    input_features = extract_features(input_image_path)
    input_features_normalized = scaler.transform([input_features])  # Normalize input image
    distances, indices = knn.kneighbors(input_features_normalized)

    similar_images = []
    threshold_distance = 1000  # Adjust this value based on results
    for i, index in enumerate(indices[0]):
        distance = distances[0][i]
        if images[index] != os.path.basename(input_image_path) and distance < threshold_distance:
            similar_images.append(images[index])

    return similar_images

# Route to serve images from the dataset
@app.route('/static/<path:filename>')
def dataset_file(filename):
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        similar_images = find_similar_images(file_path)
        
        return jsonify(similar_images=similar_images)  # Return JSON response

    return render_template('index.html')

@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/p2')
def p2():
    return render_template('p2.html')

@app.route('/p3')
def p3():
    return render_template('p3.html')

@app.route('/p4')
def p4():
    return render_template('p4.html')
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9080)  # 'host=0.0.0.0' allows access on the local network
