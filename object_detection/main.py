# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import win32gui, win32ui, win32con, win32api
from win32api import GetSystemMetrics
import time
import cv2
import keys

windowX = 0
windowY = 0
windowWidth = 800
windowHeight = 600

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
	keyList.append(char)

def keys_to_output():
	#[A, W, S, D]

	pressed = []
	for key in keyList:
		if win32api.GetAsyncKeyState(ord(key)):
			pressed.append(key)

	return pressed

def get_screenshot(hwnd):
	region = win32gui.GetWindowRect(hwnd)
	windowX = region[0]
	windowY = region[1]
	width = region[2] - windowX - 10
	height = region[3] - windowY - 35

	wDC = win32gui.GetWindowDC(hwnd)
	dcObj=win32ui.CreateDCFromHandle(wDC)
	cDC=dcObj.CreateCompatibleDC()
	dataBitMap = win32ui.CreateBitmap()
	dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
	cDC.SelectObject(dataBitMap)
	cDC.BitBlt((0,0),(width, height) , dcObj, (5,30), win32con.SRCCOPY)

	signedIntsArray = dataBitMap.GetBitmapBits(True)
	img = np.frombuffer(signedIntsArray, dtype='uint8')
	img.shape = (height, width, 4)

	# Free Resources
	dcObj.DeleteDC()
	cDC.DeleteDC()
	win32gui.ReleaseDC(hwnd, wDC)
	win32gui.DeleteObject(dataBitMap.GetHandle())

	return img

# gets all open windows and adds them in winlist
winlist = []
def enum_cb(hwnd, results):
	winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, [])

# grabs hwnd for the window named 'Grand Theft Auto V'
results = [(hwnd, title) for hwnd, title in winlist if 'Grand Theft Auto V' in title]
if(len(results)==0):
	print("Open GTA first")
	exit()

gta = results[0]
hwnd = gta[0]

# sets the window to foreground and sets its dimensions (game must run in windowed mode first)
try:
	win32gui.SetForegroundWindow(hwnd)
except:
	pass
win32gui.MoveWindow(hwnd, windowX, windowY, windowWidth, windowHeight, True)

dc = win32gui.GetDC(0)
dcObj = win32ui.CreateDCFromHandle(dc)
monitor = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))

# load the COCO class labels our YOLO model was trained on
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

last_time = time.time()
keys = keys.Keys({})

while True:
	orig_img = get_screenshot(hwnd)
	image_np = cv2.cvtColor(orig_img, cv2.COLOR_RGBA2RGB)

	#rect = win32gui.GetWindowRect(hwnd)
	#windowX = rect[0]
	#windowY = rect[1]
	#windowWidth = rect[2] - x
	#windowHeight = rect[3] - y

	print(round(1/(time.time() - last_time)))
	last_time = time.time()

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(image_np, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([windowWidth, windowHeight, windowWidth, windowHeight])
				(centerX, centerY, width, height) = box.astype("int")

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([int(centerX), int(centerY), int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	vehicle_dict = {}

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			centerX, centerY, width, height = boxes[i]

			# use the center (x, y)-coordinates to derive the top
			# and and left corner of the bounding box
			boxX = int(centerX - (width / 2))
			boxY = int(centerY - (height / 2))

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image_np, (boxX, boxY), (boxX + width, boxY + height), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(image_np, text, (boxX, boxY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			if(LABELS[classIDs[i]]=="person"):
				dcObj.DrawFocusRect((windowX + boxX, windowY + boxY, boxX + width, boxY + height))

				apx_distance = (0.5*windowWidth-centerX)**2 + (0.5*windowHeight-centerY)**2
				vehicle_dict[apx_distance] = [centerX, centerY]

		if len(vehicle_dict) > 0:
			closest = sorted(vehicle_dict.keys())[0]
			chosen_human = vehicle_dict[closest]

			x_move = 0.5*windowWidth-chosen_human[0]
			y_move = 0.5*windowHeight-chosen_human[1]

			if 'X' in keys_to_output():
				keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, -1*int(x_move), -1*int(y_move)))
	#			keys.directMouse(buttons=keys.mouse_lb_press)
	#		else:
	#			keys.directMouse(buttons=keys.mouse_lb_release)
	#	else:
	#		keys.directMouse(buttons=keys.mouse_lb_release)

	cv2.imshow('window',image_np)
	if cv2.waitKey(1) & 0Xff == ord('q'):
		cv2.destroyAllWindows()
		break