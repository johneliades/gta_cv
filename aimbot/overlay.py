"""Drawing the HUD straight over the game window (no separate preview window).

Same technique as draft_sensei's in-game overlay (sensei/glow_box.py): a
borderless, click-through, always-on-top layered window that never takes
focus, painted with per-pixel alpha via UpdateLayeredWindow. Here it covers
the game's client area exactly and shows a BGRA canvas that the HUD is drawn
on with OpenCV (see hud.py): untouched pixels stay fully transparent.

The capture copies from the SCREEN (see capture.ScreenCapture), so the overlay
would otherwise end up in the frames fed to the detector. It is excluded from
capture with SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) - Windows 10
2004 or later - so it is still shown on the monitor but not in screen grabs.
"""

import ctypes
from ctypes import wintypes

import numpy as np
import win32api
import win32con
import win32gui

TOPMOST_EVERY = 30  # re-assert always-on-top every N frames (the game can push it under)


class _BLENDFUNCTION(ctypes.Structure):
	_fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
		("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class _BITMAPINFOHEADER(ctypes.Structure):
	_fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
		("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
		("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
		("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
		("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
		("biClrImportant", wintypes.DWORD)]


# Own DLL handles, so the signatures set here don't affect other ctypes users.
_user32 = ctypes.WinDLL("user32")
_gdi32 = ctypes.WinDLL("gdi32")
_user32.GetDC.argtypes = [wintypes.HWND]
_user32.GetDC.restype = wintypes.HDC
_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_user32.UpdateLayeredWindow.argtypes = [
	wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT), ctypes.POINTER(wintypes.SIZE),
	wintypes.HDC, ctypes.POINTER(wintypes.POINT), wintypes.COLORREF,
	ctypes.POINTER(_BLENDFUNCTION), wintypes.DWORD]
_user32.UpdateLayeredWindow.restype = wintypes.BOOL
_user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
_user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.CreateDIBSection.argtypes = [
	wintypes.HDC, ctypes.POINTER(_BITMAPINFOHEADER), wintypes.UINT,
	ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
_gdi32.CreateDIBSection.restype = wintypes.HBITMAP
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteDC.argtypes = [wintypes.HDC]

_ULW_ALPHA = 2
_AC_SRC_OVER, _AC_SRC_ALPHA = 0, 1
_WDA_EXCLUDEFROMCAPTURE = 0x11


def new_canvas(width, height):
	"""A fully transparent BGRA canvas to draw the HUD on."""
	return np.zeros((height, width, 4), dtype=np.uint8)


class GameOverlay:
	"""A transparent window over the game's client area showing a BGRA canvas.

	Canvas pixels are premultiplied BGRA: drawing opaque (b, g, r, 255) colours
	onto a zeroed canvas with OpenCV gives exactly that, anti-aliased edges
	included.
	"""

	CLASS_NAME = "GtaCvOverlay"

	def __init__(self):
		self.instance = win32api.GetModuleHandle(None)
		self.class_name = f"{self.CLASS_NAME}{id(self)}"
		wc = win32gui.WNDCLASS()
		wc.hInstance = self.instance
		wc.lpszClassName = self.class_name
		wc.lpfnWndProc = lambda hwnd, msg, wp, lp: win32gui.DefWindowProc(hwnd, msg, wp, lp)
		win32gui.RegisterClass(wc)
		ex_style = (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST
			| win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE)
		self.hwnd = win32gui.CreateWindowEx(
			ex_style, self.class_name, "GTA CV Overlay", win32con.WS_POPUP,
			0, 0, 1, 1, 0, 0, self.instance, None)

		# Keep the overlay out of the screen capture (it would detect its own boxes).
		self.excluded_from_capture = bool(
			_user32.SetWindowDisplayAffinity(self.hwnd, _WDA_EXCLUDEFROMCAPTURE))

		self.size = None
		self.mem_dc = self.bitmap = self.old_bitmap = self.bits = None
		self.visible = False
		self.frames = 0

	def _prepare(self, width, height):
		"""A pixel buffer for this canvas size (recreated only when the size changes)."""
		if self.size == (width, height):
			return
		self._free_bitmap()
		header = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), width, -height, 1, 32, 0,
			0, 0, 0, 0, 0)  # negative height: top-down rows, like a numpy image
		screen_dc = _user32.GetDC(None)
		self.mem_dc = _gdi32.CreateCompatibleDC(screen_dc)
		self.bits = ctypes.c_void_p()
		self.bitmap = _gdi32.CreateDIBSection(screen_dc, ctypes.byref(header), 0,
			ctypes.byref(self.bits), None, 0)
		_user32.ReleaseDC(None, screen_dc)
		self.old_bitmap = _gdi32.SelectObject(self.mem_dc, self.bitmap)
		self.size = (width, height)

	def show(self, canvas, left, top):
		"""Show `canvas` (H x W x 4 BGRA uint8) with its top-left at screen (left, top)."""
		height, width = canvas.shape[:2]
		self._prepare(width, height)
		canvas = np.ascontiguousarray(canvas)
		ctypes.memmove(self.bits, canvas.ctypes.data, canvas.nbytes)
		position = wintypes.POINT(left, top)
		size = wintypes.SIZE(width, height)
		source = wintypes.POINT(0, 0)
		blend = _BLENDFUNCTION(_AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA)
		_user32.UpdateLayeredWindow(self.hwnd, None, ctypes.byref(position), ctypes.byref(size),
			self.mem_dc, ctypes.byref(source), 0, ctypes.byref(blend), _ULW_ALPHA)
		if not self.visible:
			win32gui.ShowWindow(self.hwnd, win32con.SW_SHOWNOACTIVATE)
			self.visible = True
		self.frames += 1
		if self.frames % TOPMOST_EVERY == 0:
			self.keep_on_top()

	def hide(self):
		if self.visible:
			win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)
			self.visible = False

	def keep_on_top(self):
		try:
			win32gui.SetWindowPos(
				self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
				win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
		except win32gui.error:
			pass

	def pump(self):
		"""Handle the window's messages (call every frame)."""
		win32gui.PumpWaitingMessages()

	def _free_bitmap(self):
		if self.mem_dc:
			_gdi32.SelectObject(self.mem_dc, self.old_bitmap)
			_gdi32.DeleteObject(self.bitmap)
			_gdi32.DeleteDC(self.mem_dc)
			self.mem_dc = self.bitmap = None
			self.size = None

	def close(self):
		self._free_bitmap()
		win32gui.DestroyWindow(self.hwnd)
		try:
			win32gui.UnregisterClass(self.class_name, self.instance)
		except win32gui.error:
			pass
