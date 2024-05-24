import cv2
import numpy as np
from controllers.HelperFunctions import *
from faceFeatures import FaceFeaturesExtractor


class VideoProcessor:
    def __init__(self, detector, n_frames=None, batch_size=60, extract_faces=False, socketio=None, session_id=None):
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.socketio = socketio
        self.session_id = session_id
        self.face_features_extractor = FaceFeaturesExtractor()
        self.extract_faces = extract_faces

    def __call__(self, filename):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('video length:', v_len)

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        faces = []
        processing_start(self.socketio, self.session_id, 'start frames processing')
        frames_count = 0
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue

                if self.extract_faces:

                    extractedFaces = self.face_features_extractor.extract_faces(frame)
                    if len(extractedFaces) == 0:
                        frames_count += 1
                        faces.append(frame)

                    if len(extractedFaces) == 1:
                        frames_count += 1
                        faces.append(extractedFaces[0])
                    else:
                        for face in extractedFaces:
                            frames_count += 1
                            faces.append(face)
                else:
                    frames_count += 1
                    faces.append(frame)

                if j == sample[-1]:
                    break

        v_cap.release()
        processing_start(self.socketio, self.session_id, 'finishing frames processing')
        return faces
