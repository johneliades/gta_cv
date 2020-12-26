import numpy as np
import cv2
import win32gui, win32ui, win32con
import time
from lane_detection import *

x = 0
y = 0
w = 800
h = 600

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

	init_lane_detection(w, h, True)

	# captures and shows each frame
	while True:
		orig_img = get_screenshot(hwnd)
		orig_img = lane_detection(orig_img)
		frame = cv2.cvtColor(orig_img, cv2.COLOR_RGBA2RGB)
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0Xff == ord('q'):
			cv2.destroyAllWindows()
			break

		#print(round(1/(time.time() - last_time)))
		last_time = time.time()

main()