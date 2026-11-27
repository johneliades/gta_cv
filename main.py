# coding: utf-8
# Object Detection GTA V Aimbot
# License: Apache License 2.0
# Uses YOLOv4-tiny for real-time person detection and aiming assistance.
#
# Entry point: python main.py   (settings live in config.py)

import ctypes
import os
import subprocess
import sys
import time
import traceback

import aimbot  # registers CUDA DLL folder before cv2 is imported
import cv2
import win32gui

import config
from aimbot import capture, hud, targeting
from aimbot.controls import KeyPresses, Mouse, is_key_held
from aimbot.detector import Detector
from aimbot.overlay import GameOverlay, new_canvas
from aimbot.self_id import SelfIdentifier


def self_status_text(self_id):
	"""One HUD line on how the own character is being identified."""
	if not config.SELF_EXCLUSION_ENABLED:
		return None
	if self_id.selecting:
		return "Self: selecting..."
	how = "picked by hand" if self_id.pinned else "auto"
	profile = "colour profile saved" if self_id.profile is not None else "no colour profile"
	return "Self: {}, {} ({} to pick yourself)".format(how, profile, config.SELF_SELECT_NEXT_KEY)


ELEVATED_FLAG = "--elevated"  # marks the copy restarted as administrator


def relaunch_as_admin():
	"""Restart this script as administrator (Windows asks for permission).

	Games that run as administrator (e.g. GTA IV) hide their key presses from
	programs that don't, so the aim, quit and selection keys would not work.
	Returns True if an elevated copy was started and this one should exit."""
	if not config.RUN_AS_ADMIN or ctypes.windll.shell32.IsUserAnAdmin() or ELEVATED_FLAG in sys.argv:
		return False
	script = os.path.abspath(sys.argv[0])
	params = subprocess.list2cmdline([script] + sys.argv[1:] + [ELEVATED_FLAG])
	shell_execute = ctypes.windll.shell32.ShellExecuteW
	shell_execute.restype = ctypes.c_void_p
	result = shell_execute(None, "runas", sys.executable, params, os.path.dirname(script), 1)
	if (result or 0) <= 32:  # declined, or failed
		print("[WARN] Not running as administrator: keys pressed in games that run as "
			"administrator (e.g. GTA IV) can't be read.")
		return False
	print("[INFO] Restarted as administrator in a new window.")
	return True


def main():
	if relaunch_as_admin():
		return
	capture.enable_dpi_awareness()

	# Taken first, while the terminal the script was run from is still in front
	ignored = capture.launch_windows()

	detector = Detector(config.MODEL_CFG, config.MODEL_WEIGHTS, config.CLASSES_FILE,
		config.USE_CUDA, config.BLOB_SIZE, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
	if config.HEAD_CLASS in detector.labels:
		print("[INFO] Aiming at detected '{}' boxes inside each person.".format(config.HEAD_CLASS))
	else:
		print("[INFO] Model has no '{}' class: aiming at the head estimated from each "
			"person box (HEAD_Y_FRACTION).".format(config.HEAD_CLASS))
	mouse = Mouse()
	screen = capture.ScreenCapture()

	# The HUD is drawn straight over the game. The capture reads from the
	# screen, so the overlay must be excluded from it (or it detects its own boxes).
	overlay = GameOverlay()
	if not overlay.excluded_from_capture:
		print("[WARN] Could not exclude the overlay from screen capture (needs Windows 10 "
			"2004+). Its boxes may be fed back into the detector.")

	ignored.add(overlay.hwnd)

	print("[INFO] Aimbot running. Hold '{}' to aim. Hold '{}' (or Ctrl+C) to quit.".format(
		config.AIM_ACTIVATION_KEY, config.QUIT_KEY))
	print("[INFO] Targeting the active window: switch to the one you want.")
	if config.SELF_EXCLUSION_ENABLED:
		print("[INFO] Press '{}' to pick your own character on screen (again for next, '{}' to pick).".format(
			config.SELF_SELECT_NEXT_KEY, config.SELF_SELECT_CONFIRM_KEY))

	self_id = SelfIdentifier()
	target_lock = targeting.TargetLock()
	keys = KeyPresses()

	last_time = time.time()
	fps_smooth = 0.0
	last_hwnd = None

	try:
		while not is_key_held(config.QUIT_KEY):
			overlay.pump()
			hwnd = capture.active_window(ignored)

			if hwnd != last_hwnd:
				last_hwnd = hwnd
				self_id.reset()  # tracks and camera motion belong to the old window
				if hwnd is not None:
					print("[INFO] Target window: '{}'".format(win32gui.GetWindowText(hwnd)))

			frame = screen.grab(hwnd) if hwnd is not None else None
			if frame is None:  # no target (terminal/desktop active) or minimised
				mouse.set_firing(False)
				self_id.reset()
				overlay.hide()
				time.sleep(0.01)
				continue

			image = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
			frame_h, frame_w = image.shape[:2]
			canvas = new_canvas(frame_w, frame_h)  # transparent; the HUD is drawn on it

			# FPS with smoothing
			now = time.time()
			dt = now - last_time
			fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt if dt > 0 else 0)
			last_time = now

			detections = detector.detect(image)
			people = [d for d in detections if d.label == config.TARGET_CLASS]
			own, self_scores = self_id.update(image, people)
			score_of = {id(d): s for d, s in zip(people, self_scores)}

			# Picking yourself by hand (see SELF_SELECT_* in config.py)
			if config.SELF_EXCLUSION_ENABLED:
				if keys.pressed(config.SELF_SELECT_NEXT_KEY):
					self_id.select_next()
				if keys.pressed(config.SELF_SELECT_CONFIRM_KEY):
					own = self_id.confirm_selection(image) or own
			enemies = [d for d in people if d is not own]

			for det in detections:
				is_person = det.label == config.TARGET_CLASS
				if det is own:
					color, label = hud.SELF_COLOR, "self (ignored)"
				else:
					color = detector.color_for(det)
					label = "{}: {:.2f}".format(det.label, det.confidence)
				hud.draw_detection(canvas, det, color, label,
					show_geometry=config.SHOW_SELF_DEBUG and is_person,
					score=score_of.get(id(det)))
			if self_id.selecting:
				hud.draw_selection(canvas, self_id.candidate, self_id.cancel_highlighted)

			# Aiming pauses while picking yourself, so you can't shoot the wrong box
			aim_active = is_key_held(config.AIM_ACTIVATION_KEY) and not self_id.selecting
			should_fire = False

			heads = [d for d in detections if d.label == config.HEAD_CLASS]
			picked = target_lock.pick(enemies, heads, frame_w, frame_h, self_id, aim_active)
			if picked is not None:
				_, head = picked
				aim_x, aim_y = targeting.box_centre(head)
				hud.draw_target_marker(canvas, head)

				if aim_active:
					# Move towards the target's head, damped for less jerky movement
					dx = int((aim_x - 0.5 * frame_w) * config.AIM_SMOOTHING)
					dy = int((aim_y - 0.5 * frame_h) * config.AIM_SMOOTHING)
					mouse.move(dx, dy)

					if config.AUTO_FIRE:
						# Only on the head, not while still crossing the body
						should_fire = targeting.crosshair_inside(head, frame_w, frame_h)

			# Press/release only on state change; releases when target lost or key let go
			mouse.set_firing(should_fire)

			hud.draw_status(canvas, fps_smooth, len(detections), len(enemies), aim_active,
				self_status_text(self_id))

			# Exactly over the client area the frame came from
			try:
				left, top, _, _ = capture.client_rect(hwnd)
			except win32gui.error:  # closed during this frame
				overlay.hide()
				continue
			overlay.show(canvas, left, top)
	except KeyboardInterrupt:
		pass
	finally:
		# Never leave the left button stuck down (quit, Ctrl+C or crash)
		mouse.set_firing(False)
		self_id.close()  # keep the learned colour profile for next time
		overlay.close()
		screen.close()


if __name__ == "__main__":
	try:
		main()
	except Exception:
		traceback.print_exc()
		if ELEVATED_FLAG not in sys.argv:
			sys.exit(1)
	finally:
		# The administrator copy has its own window, which would close at once
		if ELEVATED_FLAG in sys.argv:
			input("\nPress Enter to close this window.")
