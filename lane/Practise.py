from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
model1=load_model('model.h5')
cap=cv.VideoCapture("lane_vgt.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (192,192))
    frame = frame / 255
    frame = cv.resize(frame,(192,192))
    frame = np.array(frame)
    frame = frame[None, :, :, :]
    mask = model1.predict(frame)
    rem = np.resize(mask, (192, 192, 1))
    rem = rem.astype('float64')
    rem = rem * 255
    rem=np.where(rem>3,255,rem)
    cv.imshow('frame', rem)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
