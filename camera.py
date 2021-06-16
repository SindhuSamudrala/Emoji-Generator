import cv2
from model import FacialExpressionModel
import numpy as np
import h5py

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        # zero value in VideoCapture will take live webcam video
        # Give a video link if needed
        #self.video = cv2.VideoCapture('/Users/nilam/Desktop/Project/videos/facial_exp.mkv')
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            alpha = 0.5
            image = model.predict_emotion_image(pred)
            #resized_image = cv2.resize(image, (75, 75))
            # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
            added_image = cv2.addWeighted(fr[100:323, 100:323, :], alpha, image[0:223, 0:223, :],
                                          1 - alpha, 0)
            fr[100:323, 100:323] = added_image

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
