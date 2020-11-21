import numpy as np
import cv2
import win32gui, win32ui, win32con
import time
import math
from send_input import *
import statistics as st

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
x = 0
y = 0
w = 800
h = 600
win32gui.SetForegroundWindow(hwnd)
win32gui.MoveWindow(hwnd, x, y, w, h, True)

def draw_lines(orig_img, proc_img):
	lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, np.array([]), 40, 15)
	
	def reject_outliers(data, m=2):
		return data[abs(data - np.mean(data)) < m * np.std(data)]

	def average_lane(lane_data):
		x1s = []
		y1s = []
		x2s = []
		y2s = []
		for data in lane_data:
			x1s.append(data[0])
			y1s.append(data[1])
			x2s.append(data[2])
			y2s.append(data[3])

#		x1s = reject_outliers(np.array(x1s))
#		y1s = reject_outliers(np.array(y1s))
#		x2s = reject_outliers(np.array(x2s))
#		y2s = reject_outliers(np.array(y2s))

		return (st.mean(x1s), st.mean(y1s), st.mean(x2s), st.mean(y2s)) 

	m_pos = []
	m_neg = []

	try:
		ys = []
		for line in lines:
			x1,y1,x2,y2 = line[0]
			ys += [y1, y2]

		min_y = min(ys)
		max_y = h

		for line in lines:
			x1,y1,x2,y2 = line[0]
		
			if(math.sqrt((x1-x2)**2 + (y1-y2)**2)<200):
				continue

			m = (y2-y1)/(x2-x1)
			b = y1-m*x1

			x1 = int((min_y - b)/m)
			x2 = int((max_y - b)/m)

			if(m<0):
				m_neg.append((x1, min_y, x2, max_y))
			else:
				m_pos.append((x1, min_y, x2, max_y))
	except Exception as e:
		print(str(e))
		pass

	# ready to group the slopes together and then choose the 2 most common then average for them

	try:
		pos_cords = average_lane(m_pos)
		x1, y1, x2, y2 = pos_cords
		cv2.line(orig_img, (x1,y1), (x2,y2), (255, 255, 0), 5)
	except Exception as e:
		pass

	try:
		neg_coords = average_lane(m_neg)
		x1, y1, x2, y2 = neg_coords
		cv2.line(orig_img, (x1,y1), (x2,y2), (0, 0, 255), 5)
	except Exception as e:
		pass


def roi(proc_img):
	vertices = [np.array([[0, h],
						[0, 1.2*h/2],
						[3*w/8, h/3],
						[5*w/8, h/3],
						[w, 1.2*h/2],
						[w, h],
						], np.int32)]

	mask = np.zeros_like(proc_img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(proc_img, mask)
	return masked

def process_image(orig_img):
	# convert to gray
	proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

	# edge detection
	proc_img = cv2.Canny(proc_img, threshold1 = 300, threshold2 = 400)

	# range of interest
	proc_img = roi(proc_img)

	# apply blur to fix aliasing
	proc_img = cv2.GaussianBlur(proc_img, (5,5), 0)

#	draw_lines(proc_img, proc_img)
#	return proc_img

	draw_lines(orig_img, proc_img)
	return orig_img

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

# captures and shows each frame
last_time = time.time()
while True:
	orig_img = get_screenshot(hwnd)
	frame = cv2.cvtColor(process_image(orig_img), cv2.COLOR_RGBA2RGB)
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0Xff == ord('q'):
		cv2.destroyAllWindows()
		break

	print(round(1/(time.time() - last_time)))
	last_time = time.time()
