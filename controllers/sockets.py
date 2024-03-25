from flask_socketio import send, emit, join_room, leave_room
import os

from _thread import start_new_thread
from threading import Lock
import uuid

lock = Lock()

from Detector import predict
from app import socketio, app, request

# Dictionary to store session IDs and corresponding rooms
clients = {}

# load from .env file
UPLOAD_FOLDER = os.getenv('UPLOAD_VIDEOS_PATH')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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


@socketio.on('process_video')
def process_video(data):
    session_id = request.sid
    # Add session ID to clients dictionary with a new room
    # if session_id in clients:
    #     #break the connection
    #     leave_room(session_id)
    #     del clients[session_id]
    #     return

    clients[session_id] = session_id
    join_room(session_id)

    lock.acquire()
    start_new_thread(process_video_thread, (socketio, data, clients[session_id]))


def process_video_thread(socketio, data, session_id):
    if not session_id:
        return
    socketio.emit('processing_progress',
                  {'data': 'Starting new thread for processing video'}, room=session_id)
    n_frames = data['frames']
    if n_frames:
        n_frames = int(data['frames'])
    else:
        n_frames = 60
    video_bytes = data['video']

    if video_bytes:
        generated_unique_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_FOLDER, f"{generated_unique_id}.mp4")
        with open(video_path, 'wb') as f:
            f.write(video_bytes)
        print(f"Video saved to: {video_path}")
        print(f"Number of frames: {n_frames}")

        result, confidence, result_video_path = predict(video_path, n_frames=n_frames,
                                                        result_file_name=generated_unique_id, socketio=socketio,
                                                        session_id=session_id)

        # Emit a response back to the client
        socketio.emit('processing_response',
                      {'data': 'video processed', 'result': result, 'confidence': confidence,
                       'result_video_path': result_video_path}, room=session_id)
        lock.release()
        # Remove session ID from clients dictionary
        del clients[session_id]
        print("Thread finished")


def init():
    print("Initializing sockets controller")
