from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Ensure category folders exist
folders = ['saved_data/Female', 'saved_data/Male', 'saved_data/Animals', 'saved_data/Plants']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Initialize camera as None initially
camera = None

def preprocess_image(frame):
    """
    Preprocess the captured image:
    1. Convert to grayscale
    2. Resize to 64x64
    3. Normalize pixel values (0-1)
    4. Expand dimensions to match model input shape
    5. Convert grayscale to 3-channel RGB format
    6. Convert back to 8-bit integer format
    7. Add batch dimension if needed
    """
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 64x64 pixels
    resized_frame = cv2.resize(gray_frame, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Normalize (convert pixel values to range 0-1)
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    
    # Expand dimensions to add a single grayscale channel
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)
    
    # Convert grayscale to 3 channels (Duplicate grayscale values across RGB channels)
    rgb_frame = np.concatenate([expanded_frame] * 3, axis=-1)
    
    # Convert back to 8-bit unsigned integer format (0-255)
    rgb_frame = (rgb_frame * 255).astype(np.uint8)
    
    # Add batch dimension (useful for models requiring batch input)
    batch_frame = np.expand_dims(rgb_frame, axis=0)
    
    return batch_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Failed to access the camera'})
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already running'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/video_feed')
def video_feed():
    global camera
    if camera is None or not camera.isOpened():
        return jsonify({'status': 'error', 'message': 'Camera is not running'})

    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global camera
    category = request.form.get('category')

    if not category or category not in ["Female", "Male", "Animals", "Plants"]:
        return "Invalid category selected"

    if camera is None or not camera.isOpened():
        return "Error: Camera is not running"

    success, frame = camera.read()
    if success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"saved_data/{category}/image_{timestamp}.jpg"

        # Apply preprocessing
        processed_frame = preprocess_image(frame)

        # Save the preprocessed image
        cv2.imwrite(save_path, processed_frame[0])  # Extract single image from batch

        return f"Image saved in {category} folder (Preprocessed: Grayscale, 64x64, RGB, Normalized)"

    return "Capture failed"

if __name__ == '__main__':
    app.run(debug=True)
