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

'''from tensorflow.keras.models import load_model
from IPython.display import clear_output
import numpy as np
from google.colab.patches import cv2_imshow
from skimage import img_as_ubyte
import cv2 as cv
model1=load_model('/content/drive/MyDrive/model.h5')

def roi(f):
    h=192
    w=192
    reg=np.array([[(0,h),(0,h//2),(w,h//2),(w,h)]],np.int32)
    maskx=np.zeros_like(f)
    cv.fillPoly(maskx,reg,255)
    img=cv.bitwise_and(f,maskx)
    return img

cap=cv.VideoCapture("/content/drive/MyDrive/lane_vgt.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (192, 192))
    frame = cv.copyMakeBorder(frame, 0, 0, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))
    frame = frame / 255
    frame1 = np.array(frame)
    frame1 = frame1[None,:, :, :]
    mask1 = model1.predict(frame1)
    rem = np.resize(mask1, (192, 192))
    rem = rem * 255
    rem = np.where(rem > 3, 255, rem)
    crop=roi(rem)
    cv2_imshow(crop)
    clear_output(wait=True)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()'''