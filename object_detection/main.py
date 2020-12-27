# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import win32gui, win32ui, win32con, win32api
import time
import cv2
import keys

# ## Object detection imports
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
MODEL_PATH = ".\\object_detection\\model\\"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + MODEL_NAME + '/frozen_inference_graph.pb'

# ## Download Model
# opener = urllib.request.URLopener()
# opener.retrieve("http://download.tensorflow.org/models/object_detection/" + MODEL_FILE, MODEL_PATH + MODEL_FILE)
# tar_file = tarfile.open(MODEL_PATH + MODEL_FILE)
# for file in tar_file.getmembers():
# 	file_name = os.path.basename(file.name)
# 	if 'frozen_inference_graph.pb' in file_name:
# 		tar_file.extract(file, MODEL_PATH)

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(".\\object_detection\\data\\mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

x = 0
y = 0
w = 800
h = 600

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
	x = region[0]
	y = region[1]
	width = region[2] - x - 10
	height = region[3] - y - 35

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
win32gui.MoveWindow(hwnd, x, y, w, h, True)

last_time = time.time()
keys = keys.Keys({})

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
paused = False

with detection_graph.as_default():
	with tf.compat.v1.Session(graph=detection_graph) as sess:
		while True:
			orig_img = get_screenshot(hwnd)
			image_np = cv2.cvtColor(orig_img, cv2.COLOR_RGBA2RGB)

			#print(round(1/(time.time() - last_time)))
			last_time = time.time()
		
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run(
			  [boxes, scores, classes, num_detections],
			  feed_dict={image_tensor: image_np_expanded})
			# Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=5)

			boxes = boxes[0]
			classes = classes[0]
     
			vehicle_dict = {}
			
			if 'X' in keys_to_output():
				if paused:
					paused = False
				else:
					paused = True

			if(paused):
				cv2.putText(image_np, 'Aimbot Paused!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
			else:
				cv2.putText(image_np, 'Aimbot Working!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

			for i,b in enumerate(boxes):
				#if classes[i] == 3 or classes[i] == 6 or classes[i] == 8:
				if classes[i] == 1:
					if scores[0][i] >= 0.5:
						mid_x = (boxes[i][1] + boxes[i][3])/2
						mid_y = (boxes[i][0] + boxes[i][2])/2
						apx_distance = round(((1 - (boxes[i][3] - boxes[i][1]))**6), 2)
						cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*w)-15, 
							int(mid_y*h)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

						if(apx_distance>0.2):
							vehicle_dict[apx_distance] = [mid_x, mid_y]

			if len(vehicle_dict) > 0:
				closest = sorted(vehicle_dict.keys())[0]
				vehicle_choice = vehicle_dict[closest]

				x_move = 0.5-vehicle_choice[0]
				y_move = 0.5-vehicle_choice[1]

				if not paused:
					keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, -1*int(x_move*w), -1*int(y_move*h)))
					keys.directMouse(buttons=keys.mouse_lb_press)
			elif(not paused):
				keys.directMouse(buttons=keys.mouse_lb_release)		

		#		hm_x = x_move/0.5
		#		hm_y = y_move/0.5
		#		keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, -1*int(hm_x*w), -1*int(hm_y*h)))
			
		#		if closest < 0.1:
		#			keys.directKey("w", keys.key_release)
		#			keys.directKey("f")
		#			time.sleep(0.05)          
		#			keys.directKey("f", keys.key_release)
		#		else:
		#			keys.directKey("w")

			cv2.imshow('window',image_np)
			if cv2.waitKey(1) & 0Xff == ord('q'):
				cv2.destroyAllWindows()
				break