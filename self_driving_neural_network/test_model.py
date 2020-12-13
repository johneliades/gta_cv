# test_model.py

import numpy as np
from PIL import ImageGrab
import cv2
import time
import win32gui, win32ui, win32con, win32api
import send_input
from alexnet import alexnet

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

x = 0
y = 0
w = 800
h = 600

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
	keyList.append(char)
def keys_to_output():
	#[A, W, S, D]

	keys = []
	for key in keyList:
		if win32api.GetAsyncKeyState(ord(key)):
			keys.append(key)

	return keys

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

def main():
	model = alexnet(WIDTH, HEIGHT, LR)
	model.load(MODEL_NAME)

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
	win32gui.SetForegroundWindow(hwnd)
	win32gui.MoveWindow(hwnd, x, y, w, h, True)

	last_time = time.time()
	paused = False

	while(True):
		if not paused:
			# 800x600 windowed mode
			screen = get_screenshot(hwnd)
			#print('loop took {} seconds'.format(time.time()-last_time))
			last_time = time.time()
			screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
			screen = cv2.resize(screen, (WIDTH,HEIGHT))
			#cv2.imshow('',screen)
			prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]

			forward_thresh = 0.70
			turn_thresh = 0.75

			if(prediction[1] > forward_thresh):
				send_input.straight()
			elif(prediction[0] > turn_thresh):
				send_input.left()
			elif(prediction[2] > turn_thresh):
				send_input.right()
			else:
				send_input.straight()

		keys = keys_to_output()
		# p pauses game and can get annoying.
		if 'G' in keys:
			if paused:
				print("Unpaused Driverless")
				paused = False
				time.sleep(1)
			else:
				print("Paused Driverless")
				paused = True
				send_input.stop()
				time.sleep(1)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

main()