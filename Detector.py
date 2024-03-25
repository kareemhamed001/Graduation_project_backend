import cv2
import numpy as np
from facenet_pytorch import MTCNN
import moviepy.editor as mp
from PIL import Image
import os
import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

from keras.models import load_model, model_from_json

from faceFeatures import FaceFeaturesExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# from tensorflow_addons.optimizers import RectifiedAdam  # Import the RectifiedAdam optimizer
# from keras.utils import get_custom_objects  # Use tensorflow.keras.utils instead

# get_custom_objects().update({"RectifiedAdam": RectifiedAdam})

input_size = (256, 256)
model_path = f'E:/python/4rt grade first term/Deepfake_detection_video/last_model.pb'


# DetectionPipeline class
class DetectionPipeline:
    def __init__(self, detector, n_frames=None, batch_size=60, socketio=None, session_id=None):
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.socketio = socketio
        self.session_id = session_id
        self.face_features_extractor = FaceFeaturesExtractor()

    def __call__(self, filename):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        faces = []
        self.socketio.emit('processing_progress',
                           {'data': 'start frames processing'}, room=self.session_id)
        frames_count = 0
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frames_count += 1
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f'frame_{frames_count}.jpg', frame)
                # extracted_faces = []
                # extracted_faces = self.face_features_extractor.extract_faces(frame, save_path=None,
                #                                                              padding_y=25, padding_x=15)
                # print(f'number of faces {len(extracted_faces)}')
                # if len(extracted_faces) > 0:
                #     # faces.append(extracted_faces)
                #
                #     for face in extracted_faces:
                #         # face2 = face2 / 255
                #         # face2 = np.expand_dims(face2, axis=0)
                #         faces.append(face)
                # else:
                #     faces.append(frame)
                faces.append(frame)

                if j == sample[-1]:
                    break

        v_cap.release()
        self.socketio.emit('processing_progress',
                           {'data': 'finishing frames processing'}, room=self.session_id)
        return faces


def loadModel(model_path):
    try:
        if not os.path.exists(model_path):
            print('the model does not exist')
            raise Exception('model does not exist')

        else:
            model = load_model(model_path)
            return model
    except Exception as e:
        print('error loading the model', e)
        raise e


# Load the fold from folder fold


def predict_image(model, img):
    if img is not None:
        if input_size != (0, 0) and img.shape[0] > 0 and img.shape[1] > 0:
            img = cv2.resize(img, input_size)
            img = img / 255
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)

            predicted_class = "FAKE" if prediction < 0.4 else "REAL"
            return predicted_class, prediction

        return None, None
    else:
        return None, None


def predict(input_video, n_frames=60, result_file_name="result", socketio=None, session_id=None):
    model = loadModel(model_path)

    mtcnn = MTCNN(margin=20, keep_all=True, factor=0.7, device='cpu')
    detection_pipeline = DetectionPipeline(detector=mtcnn, n_frames=n_frames, batch_size=60, socketio=socketio,
                                           session_id=session_id)

    # check if the video exists or not
    if not os.path.exists(input_video):
        raise Exception('the video does not exist')

    faces = detection_pipeline(input_video)
    total = real = fake = 0

    socketio.emit('processing_progress',
                  {'data': 'start prediction'}, room=session_id)

    out = cv2.VideoWriter(
        f'E:/python/4rt grade first term/Deepfake_detection_video/static/results/{result_file_name}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), 1, (1280, 720))
    for face in faces:
        predicted_class, prediction = predict_image(model, face)
        if prediction is None:
            continue

        total += 1
        pred2 = prediction[0]

        if predicted_class == "FAKE":
            fake += 1
        else:
            real += 1

        face = np.array(face)
        if face.shape[0] > 0 and face.shape[1] > 0:
            face = cv2.resize(face, (1280, 720))
            out.write(face)
        socketio.emit('processing_progress',
                      {'data': 'processing frame' + str(total) + 'out of' + str(len(faces)) + ' result = ' + str(
                          (1 - pred2) * 100) + '% to be fake frame' + ' predicted class is ' + str(predicted_class)},
                      room=session_id)

    out.release()
    faces = []

    fake_ratio = fake / total
    confidence = f"{fake_ratio * 100:.2f}"

    if fake_ratio >= 0.6:
        result = "FAKE."
    else:
        result = "REAL."

    result_path = f'/static/results/{result_file_name}.mp4'
    return (result, confidence, result_path)
