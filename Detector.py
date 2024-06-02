import cv2
import numpy as np
from facenet_pytorch import MTCNN
import os
from controllers.HelperFunctions import *
from keras.models import load_model
from VideoProcessor import VideoProcessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class CNNModel:
    def __init__(self, model_path, input_size, extract_faces=False, socketio=None, session_id=None):
        self.model = self.__loadModel(model_path)
        self.socketio = socketio
        self.session_id = session_id
        self.input_size = input_size
        self.extract_faces = extract_faces

    def __loadModel(self, path):
        try:
            if not os.path.exists(path):
                print('the model does not exist')
                raise Exception('model does not exist')

            else:
                model = load_model(path)
                return model
        except Exception as e:
            print('error loading the model', e)
            raise e

    # Load the fold from folder fold

    def __predict_image(self, img):
        if img is not None:
            if self.input_size != (0, 0) and img.shape[0] > 0 and img.shape[1] > 0:
                img = cv2.resize(img, self.input_size)
                img = img / 255
                img = np.expand_dims(img, axis=0)
                prediction = self.model.predict(img)

                predicted_class = "FAKE" if prediction < 0.5 else "REAL"
                return predicted_class, prediction

            return None, None
        else:
            return None, None

    def __call__(self, input_video, n_frames=60, result_file_name="result", result_video_size=(256, 256)):

        mtcnn = MTCNN(margin=20, keep_all=True, factor=0.7, device='cpu')
        detection_pipeline = VideoProcessor(detector=mtcnn, n_frames=n_frames, batch_size=60,
                                            extract_faces=self.extract_faces, socketio=self.socketio,
                                            session_id=self.session_id)

        # check if the video exists or not
        if not os.path.exists(input_video):
            raise Exception('the video does not exist')

        faces = detection_pipeline(input_video)
        total = real = fake = 0

        processing_start(self.socketio, self.session_id, 'start prediction')

        output_extension = 'mp4'  # Specify the desired output video format (e.g., 'mp4', 'avi', etc.)
        output_file_path = f'static/results/{result_file_name}.{output_extension}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
        fps = 1
        result_video_size = (640, 480)  # Video frame size (width, height)

        # Create a VideoWriter object to save the video
        out = cv2.VideoWriter(output_file_path, fourcc, fps, result_video_size)
        frames_paths = []
        for face in faces:
            frame_name = f'{result_file_name}_{total}'
            frame_path = f'static/results/images/{frame_name}.jpg'

            try:
                print(cv2.imwrite(frame_path, face))
                frames_paths.append(frame_path)
            except Exception as e:
                print(e)

            predicted_class, prediction = self.__predict_image(face)
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
                face = cv2.resize(face, result_video_size)
                out.write(face)

            if predicted_class == 'FAKE':
                confidence = (1 - pred2) * 100
            else:
                confidence = pred2 * 100
            processing_progress(self.socketio, self.session_id, str(total), predicted_class, confidence,
                                frame_path)

        out.release()
        faces = []

        fake_ratio = fake / total
        confidence = f"{fake_ratio * 100:.2f}"

        if fake_ratio >= 0.6:
            result = "FAKE"
        else:
            result = "REAL"

        return result, confidence, output_file_path, frames_paths

    def predict_image(self, image_path, result_file_name):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at path {image_path} could not be loaded. Please check the path and file format.")

        predicted_class, prediction = self.__predict_image(img)
        if prediction is None:
            return

        if predicted_class == "FAKE":
            confidence = (1 - prediction) * 100
        else:
            confidence = prediction * 100

        processing_progress(socketObject=self.socketio, session_id=self.session_id, frame_number=str(1),
                            class_name=predicted_class, confidence=confidence,
                            frame_path=image_path)

        processing_result(socketObject=self.socketio, session_id=self.session_id, class_name=predicted_class,
                          confidence=confidence,
                          result_media_path=image_path, frames_paths=[image_path])
        return predicted_class, confidence
