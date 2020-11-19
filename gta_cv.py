import numpy as np
import cv2
import win32gui
import time
from PIL import ImageGrab

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

def grayscale(orig_img):
	proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
	proc_img = cv2.Canny(proc_img, threshold1 = 50, threshold2 = 100)

	return proc_img

# captures and shows each frame
#last_time = time.time()
while True:
	orig_img = np.array(ImageGrab.grab(bbox=(x+10,  y+30, w-10, h-10)))

	frame = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0Xff == ord('q'):
		break

#	print(round((time.time() - last_time)*1000))
#	last_time = time.time()

cv2.destroyAllWindows()