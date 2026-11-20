"""Keyboard polling and mouse output (movement + fire button)."""

import win32api

from . import keys


def is_key_held(key):
	"""True only while the key is physically down (high bit), not 'pressed since last poll'."""
	return bool(win32api.GetAsyncKeyState(ord(key.upper())) & 0x8000)


class Mouse:
	def __init__(self):
		self._keys = keys.Keys({})
		# Whether WE are holding the left button. Only edge events are sent
		# (press once when acquiring, release once when losing the target),
		# so the user's own clicks are never interfered with.
		self.firing = False

	def move(self, dx, dy):
		"""Relative mouse move, in mouse units."""
		worker = self._keys.keys_worker
		worker.SendInput(worker.Mouse(0x0001, dx, dy))

	def set_firing(self, should_fire):
		"""Press/release the left button only when the state changes."""
		if should_fire and not self.firing:
			self._keys.directMouse(buttons=self._keys.mouse_lb_press)
		elif not should_fire and self.firing:
			self._keys.directMouse(buttons=self._keys.mouse_lb_release)
		self.firing = should_fire
