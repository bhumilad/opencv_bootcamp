import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve


# ========================-Downloading Assets-========================
# def download_and_unzip(url, save_path):
#     print(f"Downloading and extracting assests....", end="")

#     # Downloading zip file using urllib package.
#     urlretrieve(url, save_path)

#     try:
#         # Extracting zip file using the zipfile package.
#         with ZipFile(save_path) as z:
#             # Extract ZIP file contents in the same directory.
#             z.extractall(os.path.split(save_path)[0])

#         print("Done")

#     except Exception as e:
#         print("\nInvalid file.", e)


# URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

# asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# # Download if assest ZIP does not exists.
# if not os.path.exists(asset_zip_path):
#     download_and_unzip(URL, asset_zip_path)
# ====================================================================



# cv2.dnn.readNetFromCaffe() is an OpenCV function that loads a Caffe deep learning model so you can use it for inference (object detection, image classification, etc.).

# cv2.dnn.readNetFromCaffe(prototxt, caffeModel)


# prototxt → The .prototxt file that defines the network architecture (layers, connections, etc.).

# caffeModel → The .caffemodel file that contains the trained weights.

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)
win_name = "Video Player"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model Parameters
# We resize the picture to 300x300

in_width = 300
in_height = 300
# When a neural network is trained, the people who trained it looked at all the images in the training dataset.

# They calculated the average color values (for Blue, Green, Red channels).

# In this case, the averages were about 104 for Blue, 117 for Green, and 123 for Red.

# So, during training, they subtracted those numbers to make the data balanced.
mean_val = (104.0, 177.0, 123.0)
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    # Create a 4D blob from a frame.
    # ----------------------------------
    #Most important preprocessing step
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean_val, swapRB = False, crop =False)
    # Run a model
    net.setInput(blob)
    # The image goes through the neural network (all the math layers).

# At the end, the network makes predictions about objects.
# detections : [batch, number_of_detections, 1, attributes]
# batch → how many images you gave at once. (Usually 1 for us, since we give one image at a time).

# number_of_detections → how many objects the model thinks it found (could be 100, even if most are junk).

# 1 → just a placeholder dimension (don’t worry much about it).

# attributes → the actual info for each detection. Usually 7 numbers:

# Image ID (helps when you give a batch, always 0 for one image)

# Class ID (what object it is: car, person, dog, etc.)

# Confidence (how sure the model is, e.g., 0.85 = 85% sure)

# x_min (left side of the box, as a % of image width)

# y_min (top side of the box, as a % of image height)

# x_max (right side of the box)

# y_max (bottom side of the box)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_top_left = int(detections[0, 0, i, 3] * frame_width)
            y_top_left = int(detections[0, 0, i, 4] * frame_height)
            x_bottom_right  = int(detections[0, 0, i, 5] * frame_width)
            y_bottom_right  = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame,  (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_top_left, y_top_left - label_size[1]),
                (x_top_left + label_size[0], y_top_left + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
# Measures how long the network took to process the frame.

# Displays inference time in milliseconds.
    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)
source.release()
cv2.destroyAllWindows()

