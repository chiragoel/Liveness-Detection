import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from collections import defaultdict
import pickle
import face_recognition

import glob
import sklearn
import skimage.io as io
import skimage.filters as flt
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern

import keras
import tensorflow as tf
from keras import datasets, layers, models, optimizers, losses, regularizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model, Sequential
from keras.layers import InputLayer, Dense, Flatten, Dropout, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation, concatenate, AveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adagrad, RMSprop, SGD, Adam
from keras.utils import to_categorical

IMG_SIZE = 24
IMG_SIZE_RF = 128

cascade_dir = "cascade_files/"
eye_state_dir = "data/eye_state/"

face_casc_path = cascade_dir + "haarcascade_frontalface_default.xml"
eye_casc_path = cascade_dir + "haarcascade_eye.xml"
open_eye_casc_path = cascade_dir + "haarcascade_eye_tree_eyeglasses.xml"
left_eye_casc_path = cascade_dir + "haarcascade_lefteye_2splits.xml"
right_eye_casc_path = cascade_dir + "haarcascade_righteye_2splits.xml"

def load_cascades():
    face_detector = cv2.CascadeClassifier(face_casc_path)
    eye_detector = cv2.CascadeClassifier(eye_casc_path)
    open_eye_detector = cv2.CascadeClassifier(open_eye_casc_path)
    left_eye_detector = cv2.CascadeClassifier(left_eye_casc_path)
    right_eye_detector = cv2.CascadeClassifier(right_eye_casc_path)

    return face_detector, eye_detector, open_eye_detector, left_eye_detector, right_eye_detector

def Eye_state_Classifier(nb_classes = 1):
    model = Sequential()
    
    model.add(Conv2D(6, (3, 3), activation = 'relu', input_shape = (IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu'))
    model.add(Dense(84, activation = 'relu'))
    model.add(Dense(nb_classes, activation = 'sigmoid'))
    
    return model

def load_pretrained_model():
            
    model = Eye_state_Classifier(nb_classes = 1)

    INIT_LR = 1e-3
    EPOCHS = 25
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) #Optimise using Adam 
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
    #model.summary()
    # load weights into new model

    model.load_weights("models/eye_status_classifier.h5")
    return model


def computeHaralick(gray):

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = ['energy', 'homogeneity']

    glcm = greycomatrix(gray,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)

    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    return feats


def computeLBP(gray):
    lbp = local_binary_pattern(gray, 24, 8, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                              bins=np.arange(0, 24 + 3),
                              range=(0, 24 + 2))
    hist = hist / sum(hist)

    return hist

def predict(img, model):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype('float32')
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255
    pred = model.predict(img)
    if pred < 0.4:
        pred = 'closed'
    elif pred > 0.6:
        pred = 'open'
    else:
        pred = 'idk'
    return pred

def predict_rf(img, model):
    img = cv2.resize(img, (IMG_SIZE_RF, IMG_SIZE_RF))
    haralick = computeHaralick(img)
    lbp = computeLBP(img)
    final_feats = np.concatenate([haralick,lbp])
    final_feats = final_feats.reshape((1,-1))
    pred = model.predict(final_feats)
    if pred < 0.4:
        pred = 'real'
    elif pred > 0.6:
        pred = 'fake'
    else:
        pred = 'idk'
    return pred


def init():
    face_detector, eye_detector, open_eye_detector, left_eye_detector, right_eye_detector = load_cascades()
    model = load_pretrained_model()
    
    images = []

    return (model, face_detector, eye_detector, open_eye_detector, left_eye_detector, right_eye_detector, images)

(model, face_detector, eye_detector, open_eye_detector, left_eye_detector, right_eye_detector, images) = init()
rf_model = pickle.load(open("models/randomforest.sav", 'rb'))

def isBlinking(history, maxFrames):
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False



def detect_and_display(model, rf_model, video_capture, face_detector, eye_detector, open_eye_detector, left_eye_detector, right_eye_detector, data, eyes_detected, prev_encoding):
    _, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces)>0:
      #Find largest
        rect = sorted(faces,reverse = True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (x, y, w, h) = rect

    # for (x,y,w,h) in faces:
        pred_rf = predict_rf(gray[y:y+h,x:x+w], rf_model)
        
        encoding = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])[0]

        if prev_encoding == []:
            prev_encoding = encoding
        
        matches = face_recognition.compare_faces([prev_encoding], encoding)
        name = "Unknown"

        if False in matches:
            eyes_detected[name] = ""            

        prev_encoding = encoding

        face = frame[y:y+h,x:x+w]
        gray_face = gray[y:y+h,x:x+w]

        eyes = []

        open_eyes_glasses = open_eye_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(open_eyes_glasses) == 2:
            eyes_detected[name]+='1'
            for (ex,ey,ew,eh) in open_eyes_glasses:
                cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        else:
            left_face = frame[y:y+h, x+int(w/2):x+w]
            left_face_gray = gray[y:y+h, x+int(w/2):x+w]

            right_face = frame[y:y+h, x:x+int(w/2)]
            right_face_gray = gray[y:y+h, x:x+int(w/2)]

            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            eye_status = '1'

            for (ex,ey,ew,eh) in right_eye:
                color = (0,255,0)
                pred = predict(right_face_gray[ey:ey+eh,ex:ex+ew],model)
                if pred == 'closed':
                    eye_status='0'
                    color = (0,0,255)
                cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
            for (ex,ey,ew,eh) in left_eye:
                color = (0,255,0)
                pred = predict(left_face_gray[ey:ey+eh,ex:ex+ew],model)
                if pred == 'closed':
                    eye_status='0'
                    color = (0,0,255)
                cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
            eyes_detected[name] += eye_status

        blink_output = isBlinking(eyes_detected[name],10)
        if blink_output or pred_rf=='real':
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, 'Real: '+name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        else:
            if len(eyes_detected[name]) > 20:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, 'Fake: '+name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), 2)


    return frame

data = {'encodings': []}

print("[LOG] Opening webcam ...")
video_capture = cv2.VideoCapture(0)

eyes_detected = defaultdict(str)

prev_encoding = []

while True:
    
  frame = detect_and_display(model, rf_model, video_capture, face_detector, eye_detector, open_eye_detector, left_eye_detector, right_eye_detector, data, eyes_detected, prev_encoding)
  cv2.imshow("Eye-Blink LiveNet", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
        
video_capture.release()
cv2.destroyAllWindows()
