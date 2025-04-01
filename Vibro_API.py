from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Define a directory to save the uploaded audio files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "Home"

@app.route("/get-user/<sound>")
def predict(sound):
    return {
        "user_id": sound
    }

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
    
    return jsonify({"message": "Audio file uploaded successfully", "file_path": file_path}), 201

if __name__ == "__main__":
    app.run(debug=True)
