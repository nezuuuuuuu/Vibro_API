from flask import Flask, request, jsonify
import os
from model import YAMNetClassifier  # Import the YAMNet class

app = Flask(__name__)
yamnet = YAMNetClassifier()

# Define a directory to save the uploaded audio files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "Home"


@app.route("/get-prediction", methods=["POST"])
def getsound():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the audio file to the 'uploads' directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(file_path)
    
    # Use YAMNet to predict the sound
    prediction = yamnet.predict(file_path)
    
    return jsonify({"message": "Audio file processed successfully", "prediction": prediction}), 200


if __name__ == "__main__":
    app.run(debug=True)
