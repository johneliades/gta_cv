import numpy as np
import cv2
import win32gui
from PIL import ImageGrab

winlist = []
def enum_cb(hwnd, results):
	winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

# gets all open windows and adds them in winlist
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
win32gui.MoveWindow(hwnd, 0, 0, 800, 600, True)

# captures and shows each frame
while True:
	img = ImageGrab.grab(bbox=(10, 30, 790, 590)) #x, y, w, h
	img_np = np.array(img)
	frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) & 0Xff == ord('q'):
		break

cv2.destroyAllWindows()