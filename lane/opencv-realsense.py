from tensorflow.keras.models import load_model
import numpy as np
from skimage import img_as_ubyte
import cv2 as cv
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

model1 = load_model('/home/piyush/Downloads/funet (1).h5')

while True:
    frames = pipeline.wait_for_frames()
    # ret, frame = cap.read()
    # if not ret:
    #    break

    frame = frames.get_color_frame()
    frame=np.asanyarray(frame.get_data())
    frame = cv.copyMakeBorder(frame, 0, 0, 25, 25, cv.BORDER_CONSTANT, value=(0, 0, 0))
    frame = frame / 255
    frame = cv.resize(frame, (192, 192))
    frame=np.expand_dims(frame,axis=-1)
    frame1 = frame[None, :, :, :]
    mask1 = model1.predict(frame1)
    rem = np.resize(mask1, (192, 192, 1))
    rem = img_as_ubyte(rem)
    rem = np.round(rem)
    rem = rem * 255
    cv.imshow("", rem)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
