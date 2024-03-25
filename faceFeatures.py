import cv2
import numpy as np


class FaceFeaturesExtractor:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier()
        self.face_cascade.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.eye_cascade = cv2.CascadeClassifier()
        self.eye_cascade.load(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.left_eye_cascade = cv2.CascadeClassifier()
        self.left_eye_cascade.load(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

        self.right_eye_cascade = cv2.CascadeClassifier()
        self.right_eye_cascade.load(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

    def extract_faces(self, image, save_path=None, padding_y=20, padding_x=10):
        # check if image is path or image
        img = self.__check_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        crop_faces = []
        faces_count = 0
        # if faces not empty
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                crop_face = img[y - padding_y:y + h + padding_y, x - padding_x:x + w + padding_x]
                # save the crop face
                if save_path and crop_face.any():
                    cv2.imwrite(f'{save_path}_{faces_count}.jpg', crop_face)
                crop_faces.append(crop_face)
                faces_count += 1

        return crop_faces

    def extrac_eyes(self, image, save_path=None, padding_y=20, padding_x=10):
        img = self.__check_image(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray)
        print(len(eyes))
        eyes_count = 0
        eyes_array = []
        for (ex, ey, ew, eh) in eyes:
            crop_eye = img[ey - padding_y:ey + eh + padding_y, ex - padding_x:ex + ew + padding_x]
            if not crop_eye.any():  # Check if crop_eye is empty
                print(f"Empty eye detected at {ex}, {ey}. Skipping...")
                continue
            if save_path:
                cv2.imwrite(f'{save_path}_{eyes_count}.jpg', crop_eye)
            eyes_count += 1
            eyes_array.append(crop_eye)

        return eyes_array

    def __check_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise ValueError('image must be a path or numpy array')
        return img
