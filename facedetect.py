# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

faceCascade =   cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def draw_boundary(img, classifier):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)
        # cv2.putText(img, "jigkoh", (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),3)
        coords = [x,y,w,h]
    return img, coords

def detect(img):
    img, coords = draw_boundary(img,faceCascade)
    return img


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
    image = detect(image)
	# show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF

	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break