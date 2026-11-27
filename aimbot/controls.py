"""Keyboard polling and mouse output (movement + fire button)."""

import ctypes
from ctypes import wintypes

import win32api
import win32con

_MOUSEEVENTF_MOVE = 0x0001       # relative move
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_INPUT_MOUSE = 0


class _MOUSEINPUT(ctypes.Structure):
	_fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
		("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUT(ctypes.Structure):
	# MOUSEINPUT is the largest member of the real INPUT union, so this has
	# the size SendInput expects.
	_fields_ = [("type", wintypes.DWORD), ("mi", _MOUSEINPUT)]


_user32 = ctypes.WinDLL("user32")
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT


def _send_mouse(flags, dx=0, dy=0):
	"""One low-level mouse event (SendInput: games reading raw input see it)."""
	event = _INPUT(_INPUT_MOUSE, _MOUSEINPUT(dx, dy, 0, flags, 0, 0))
	_user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT))


_NAMED_KEYS = {
	"ENTER": win32con.VK_RETURN,
	"SPACE": win32con.VK_SPACE,
	"TAB": win32con.VK_TAB,
	"BACKSPACE": win32con.VK_BACK,
	"SHIFT": win32con.VK_SHIFT,
	"CTRL": win32con.VK_CONTROL,
	"ALT": win32con.VK_MENU,
}


def _virtual_key(key):
	"""Virtual-key code for a letter/digit ("X"), a function key ("F10") or a
	named key ("ENTER", "SPACE", "TAB", "BACKSPACE", "SHIFT", "CTRL", "ALT")."""
	key = key.upper()
	if key in _NAMED_KEYS:
		return _NAMED_KEYS[key]
	if len(key) > 1 and key[0] == "F" and key[1:].isdigit():
		return win32con.VK_F1 + int(key[1:]) - 1
	return ord(key)


def is_key_held(key):
	"""True only while the key is physically down (high bit), not 'pressed since last poll'."""
	return bool(win32api.GetAsyncKeyState(_virtual_key(key)) & 0x8000)


class Mouse:
	def __init__(self):
		# Whether WE are holding the left button. Only edge events are sent
		# (press once when acquiring, release once when losing the target),
		# so the user's own clicks are never interfered with.
		self.firing = False

	def move(self, dx, dy):
		"""Relative mouse move, in mouse units."""
		_send_mouse(_MOUSEEVENTF_MOVE, dx, dy)

	def set_firing(self, should_fire):
		"""Press/release the left button only when the state changes."""
		if should_fire and not self.firing:
			_send_mouse(_MOUSEEVENTF_LEFTDOWN)
		elif not should_fire and self.firing:
			_send_mouse(_MOUSEEVENTF_LEFTUP)
		self.firing = should_fire


class KeyPresses:
	"""Reports each key press once (on the down edge), however long it is held."""

	def __init__(self):
		self._was_down = {}

	def pressed(self, key):
		down = is_key_held(key)
		was_down = self._was_down.get(key, False)
		self._was_down[key] = down
		return down and not was_down
