from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
from datetime import datetime

app = Flask(__name__)

# Ensure category folders exist
folders = ['saved_data/Female', 'saved_data/Male', 'saved_data/Animals', 'saved_data/Plants']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Camera state
camera = None

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
    if camera is None:
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
        cv2.imwrite(save_path, frame)
        return f"Image saved in {category} folder"

    return "Capture failed"

if __name__ == '__main__':
    app.run(debug=True)
