from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import importlib
import numpy as np
from numpy import identity
from retinaface import RetinaFace
import cv2 as cv
import matplotlib.pyplot as plt
import textwrap

model = load_model("mask_detector.model")
cap = cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    resp = RetinaFace.detect_faces(frame)
    if hasattr(resp, 'keys'):
        for key in resp.keys():
            identity = resp[key]
            facial_area = identity["facial_area"]
            face = frame[facial_area[1]:facial_area[3], facial_area[0]: facial_area[2]]
                
            face = cv.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]
            print(mask)
            print(withoutMask)
            label = "Mask" if mask > withoutMask else "No Mask"
            if label == "Mask":
                cv.putText(frame, label + " "+ np.array2string(mask), (facial_area[0], facial_area[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv.rectangle(frame,(facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (36,255,12),1)
            else:
                cv.putText(frame, label + " "+ np.array2string(withoutMask), (facial_area[0], facial_area[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv.rectangle(frame,(facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0,0,255),1)

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

