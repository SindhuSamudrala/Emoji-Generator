from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)



class FacialExpressionModel(object):

    EMOTIONS_MAPPING = {
        'Angry': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Angry.png"),
        'Disgust': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Disgust.png"),
        'Fear': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Fear.png"),
        'Happy': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Happy.png"),
        'Neutral': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Neutral.png"),
        'Sad': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Sad.png"),
        'Surprise': cv2.imread("C:/Users/Sindhu Samudrala/Documents/MSDA/DATA255/Project/Emojis/Surprise.png"),
    }
    EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def predict_emotion_image(self, pred_emo):
        return FacialExpressionModel.EMOTIONS_MAPPING.get(pred_emo)
