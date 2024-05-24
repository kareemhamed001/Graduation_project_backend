from flask_socketio import send, emit, join_room, leave_room
import os
from _thread import start_new_thread
from threading import Lock
import uuid
from Detector import CNNModel
from app import socketio, app, request
from dotenv import load_dotenv
from controllers.HelperFunctions import processing_result, processing_progress, processing_start

lock = Lock()

# Dictionary to store session IDs and corresponding rooms
clients = {}

# load from .env file
load_dotenv()
UPLOAD_FOLDER = os.getenv('UPLOAD_VIDEOS_PATH')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# model data
input_size = (256, 256)
model_path = os.getenv('MODEL_PATH')


@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    # Add session ID to clients dictionary with a new room
    clients[session_id] = session_id
    join_room(session_id)


@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    # Remove session ID from clients dictionary
    del clients[session_id]


@socketio.on('process_media')
def process_video(data):
    print('start prooccess media')
    session_id = request.sid
    clients[session_id] = session_id
    join_room(session_id)

    lock.acquire()
    start_new_thread(process_video_thread, (socketio, data, clients[session_id]))


def process_video_thread(socketio, data, session_id):
    if not session_id:
        return

    processing_start(socketio, session_id, 'Starting new thread for processing video')

    n_frames = data['frames']
    extract_faces = data['extract_faces']
    if n_frames:
        n_frames = int(data['frames'])
    else:
        n_frames = 30
    media_type = data['type']
    print(f"Media type: {media_type}")
    if media_type == 'video':
        print('video')
        video_bytes = data['media']

        if video_bytes:
            generated_unique_id = str(uuid.uuid4())
            video_path = os.path.join(UPLOAD_FOLDER, f"{generated_unique_id}.mp4")
            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            print(f"Video saved to: {video_path}")
            print(f"Number of frames: {n_frames}")

            cnnModel = CNNModel(model_path=model_path, input_size=input_size, extract_faces=extract_faces,
                                socketio=socketio, session_id=session_id)

            result, confidence, result_video_path, frames_paths = cnnModel(input_video=video_path, n_frames=n_frames,
                                                                           result_file_name=generated_unique_id)

            processing_result(socketio, session_id, result, confidence, result_video_path, frames_paths)

            lock.release()
            # Remove session ID from clients dictionary
            del clients[session_id]
            print("Thread finished")

    elif media_type == 'image':
        print('image')
        image_bytes = data['media']
        if image_bytes:
            generated_unique_id = str(uuid.uuid4())
            image_path = os.path.join(UPLOAD_FOLDER, f"{generated_unique_id}.jpg")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to: {image_path}")

            cnnModel = CNNModel(model_path=model_path, input_size=input_size, socketio=socketio,
                                session_id=session_id)

            predicted_class, confidence = cnnModel.predict_image(image_path, generated_unique_id)

            processing_result(socketio, session_id, predicted_class, confidence, image_path, image_path)
            lock.release()
            # Remove session ID from clients dictionary
            del clients[session_id]
            print("Thread finished")


def init():
    print("Initializing sockets controller")
