import numpy as np
import cv2
import win32gui, win32ui, win32con
import time
import send_input

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
	lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, np.array([]), 200, 15)

	try:
		for line in lines:
			coords = line[0]
			cv2.line(orig_img, (coords[0],coords[1]), 
								(coords[2],coords[3]),
								[0, 255, 255], 3)
	except:
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
	proc_img = cv2.Canny(proc_img, threshold1 = 150, threshold2 = 250)

	# range of interest
	proc_img = roi(proc_img)

	# apply blur to fix aliasing
	proc_img = cv2.GaussianBlur(proc_img, (5,5), 0)
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
	img = np.fromstring(signedIntsArray, dtype='uint8')
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
	orig_img = process_image(orig_img)
	frame = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2RGB)
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0Xff == ord('q'):
		cv2.destroyAllWindows()
		break

	print(round(1/(time.time() - last_time)))
	last_time = time.time()

#send_input.PressKey(send_input.W)
#send_input.ReleaseKey(send_input.W)