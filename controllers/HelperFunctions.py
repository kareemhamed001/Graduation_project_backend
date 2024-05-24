import numpy as np


def processing_start(socketObject, session_id, message):
    socketObject.emit('processing_start', {'message': message}, room=session_id)
    return True


def processing_progress(socketObject, session_id, frame_number, class_name, confidence, frame_path=None):
    if isinstance(class_name, np.ndarray):
        class_name = class_name.tolist()
    if isinstance(confidence, np.ndarray):
        confidence = confidence.tolist()
    socketObject.emit('processing_progress',
                      {'frame_number': frame_number, 'class_name': class_name, 'confidence': confidence,
                       'frame_path': frame_path},
                      room=session_id)
    return True


def processing_result(socketObject, session_id, class_name, confidence, result_media_path, frames_paths=[]):
    # Convert to JSON serializable format if necessary
    if isinstance(class_name, np.ndarray):
        class_name = class_name.tolist()
    if isinstance(confidence, np.ndarray):
        confidence = confidence.tolist()
    socketObject.emit('processing_result',
                      {'class_name': class_name, 'confidence': confidence, 'result_media_path': result_media_path,
                       'frames_paths': frames_paths},
                      room=session_id)
    return True
