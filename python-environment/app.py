from flask import Flask, request, jsonify
import asyncio
import os
from werkzeug.utils import secure_filename
from main_pipeline import run_pipeline  # Assuming your pipeline code is in main_pipeline.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/run', methods=['POST'])
def run():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Run the pipeline asynchronously
    result = asyncio.run(run_pipeline(video_path))

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
