import face_recognition
import cv2
import numpy as np
import base64
import io
import os
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
 
# Path to dataset folder
dataset_path = "E:/Projects/Police_department/police_department/Server/dataset/"

# Store all encodings and names
known_encodings = []
known_names = []

# Load all images from dataset folder
for image_file in os.listdir(dataset_path):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        # Load image and encode
        image_path = os.path.join(dataset_path, image_file)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_encodings.append(encoding[0])
            # Extract name from image filename (e.g., rahul.png -> Rahul)
            name = os.path.splitext(image_file)[0]
            known_names.append(name.capitalize())

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1]  # Remove base64 header
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect face
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if not face_encodings:
            return jsonify({"name": "No face detected"})

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                return jsonify({"name": known_names[best_match_index]})

        return jsonify({"name": "Unknown"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
