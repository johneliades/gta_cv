"""Finding the game window and capturing its frames."""

import ctypes

import numpy as np
import win32con
import win32gui
import win32ui


def enable_dpi_awareness():
	"""Make window/client coordinates real pixels.

	Without this, at Windows display scaling > 100% the rects are
	DPI-virtualised and don't match the captured pixels, which shifts the
	computed crosshair.
	"""
	try:
		ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
	except Exception:
		try:
			ctypes.windll.user32.SetProcessDPIAware()
		except Exception:
			pass


def find_window(title_substring):
	"""Return the handle of the first top-level window whose title contains
	title_substring, or None."""
	windows = []
	win32gui.EnumWindows(lambda hwnd, _: windows.append((hwnd, win32gui.GetWindowText(hwnd))), None)
	for hwnd, title in windows:
		if title_substring in title:
			return hwnd
	return None


def place_window(hwnd, x, y, width, height):
	"""Bring the window to the front and move/resize it."""
	try:
		win32gui.SetForegroundWindow(hwnd)
	except Exception:
		pass
	win32gui.MoveWindow(hwnd, x, y, width, height, True)


def grab_frame(hwnd):
	"""Capture the window's client area as a BGRA numpy array (H, W, 4).

	Captures exactly the client area (the game's render surface, no title bar
	or borders), so the image centre is the in-game crosshair. Border sizes
	vary with DPI scaling and Windows theme, so they are measured, not guessed.

	Copies from the SCREEN, not the game window's DC: GTA V renders with
	DirectX, and a BitBlt from the window's own DC returns black for DX
	swapchains on modern Windows. Requires windowed / borderless mode (not
	exclusive fullscreen) and the game not covered by other windows.

	Returns None if the window has no visible client area (e.g. minimised).
	"""
	_, _, width, height = win32gui.GetClientRect(hwnd)
	if width <= 0 or height <= 0:
		return None
	left, top = win32gui.ClientToScreen(hwnd, (0, 0))

	screen_dc = win32gui.GetDC(0)
	dc = win32ui.CreateDCFromHandle(screen_dc)
	mem_dc = dc.CreateCompatibleDC()
	bitmap = win32ui.CreateBitmap()
	try:
		bitmap.CreateCompatibleBitmap(dc, width, height)
		mem_dc.SelectObject(bitmap)
		mem_dc.BitBlt((0, 0), (width, height), dc, (left, top), win32con.SRCCOPY)
		bits = bitmap.GetBitmapBits(True)
		return np.frombuffer(bits, dtype="uint8").reshape(height, width, 4)
	finally:
		dc.DeleteDC()
		mem_dc.DeleteDC()
		win32gui.ReleaseDC(0, screen_dc)
		win32gui.DeleteObject(bitmap.GetHandle())


def window_right_edge(hwnd):
	"""Screen x coordinate of the window's right edge."""
	return win32gui.GetWindowRect(hwnd)[2]
