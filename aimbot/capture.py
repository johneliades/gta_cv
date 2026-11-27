"""Picking the target (active) window and capturing its frames."""

import ctypes

import win32gui


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


# The desktop and taskbars are "foreground" when you click on them; never a target.
_SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


def active_window(ignore=()):
	"""The foreground window if it can be a target, else None.

	None for windows in `ignore` (e.g. the terminal the script was started
	from), the desktop / taskbar, and minimised windows."""
	hwnd = win32gui.GetForegroundWindow()
	if not hwnd or hwnd in ignore or win32gui.IsIconic(hwnd):
		return None
	try:
		if win32gui.GetClassName(hwnd) in _SHELL_CLASSES:
			return None
	except win32gui.error:  # closed in the meantime
		return None
	return hwnd


def launch_windows():
	"""Windows to never target: the one in the foreground when the script
	starts (the terminal / IDE it was run from) and its console window."""
	windows = {win32gui.GetForegroundWindow(), ctypes.windll.kernel32.GetConsoleWindow()}
	return {hwnd for hwnd in windows if hwnd}


class ScreenCapture:
	"""Captures the game window's client area as a BGRA numpy array (H, W, 4).

	Captures exactly the client area (the game's render surface, no title bar
	or borders), so the image centre is the in-game crosshair. Border sizes
	vary with DPI scaling and Windows theme, so they are measured, not guessed.

	Uses DXGI Desktop Duplication (the dxcam package): reads the frame straight
	from the GPU's desktop image. Copies from the SCREEN, not the game window,
	so it needs windowed / borderless mode and the game not covered by other
	windows. Windows excluded from capture (our overlay) are left out.
	"""

	def __init__(self):
		self._outputs = _dxcam_outputs()  # [(camera, (left, top, right, bottom) on the desktop)]
		if not self._outputs:
			raise RuntimeError("dxcam found no monitor to capture")

	def grab(self, hwnd):
		"""The client area as BGRA (H, W, 4), or None if there is nothing to
		capture this frame (minimised, closed, spanning two monitors, or a
		display mode change)."""
		try:
			left, top, width, height = client_rect(hwnd)
		except win32gui.error:  # the window closed
			return None
		if width <= 0 or height <= 0:
			return None
		right, bottom = left + width, top + height
		for camera, (o_left, o_top, o_right, o_bottom) in self._outputs:
			if o_left <= left and o_top <= top and right <= o_right and bottom <= o_bottom:
				region = (left - o_left, top - o_top, right - o_left, bottom - o_top)
				try:
					# new_frame_only=False: if the screen hasn't changed since the
					# last grab, reuse that frame instead of returning None.
					return camera.grab(region=region, new_frame_only=False)
				except Exception:
					return None  # e.g. display mode change
		return None  # spans monitors / partly off-screen

	def close(self):
		for camera, _ in self._outputs:
			try:
				camera.release()
			except Exception:
				pass
		self._outputs = []


def _dxcam_outputs():
	"""A dxcam camera for every monitor, with the monitor's desktop rect."""
	import dxcam

	outputs = []
	for device_idx in range(8):
		for output_idx in range(16):
			try:
				camera = dxcam.create(device_idx=device_idx, output_idx=output_idx,
					output_color="BGRA")
			except (IndexError, KeyError):
				break  # no more outputs on this adapter (or no such adapter)
			rect = camera._output.desc.DesktopCoordinates
			outputs.append((camera, (rect.left, rect.top, rect.right, rect.bottom)))
	return outputs


def client_rect(hwnd):
	"""(left, top, width, height) of the window's client area, in screen pixels.
	The overlay is placed here, exactly over the captured frame."""
	_, _, width, height = win32gui.GetClientRect(hwnd)
	left, top = win32gui.ClientToScreen(hwnd, (0, 0))
	return left, top, width, height
