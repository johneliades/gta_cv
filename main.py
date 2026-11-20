# coding: utf-8
# Object Detection GTA V Aimbot
# License: Apache License 2.0
# Uses YOLOv4-tiny for real-time person detection and aiming assistance.
#
# Entry point: python main.py   (settings live in config.py)

import sys
import time

import aimbot  # registers CUDA DLL folder before cv2 is imported
import cv2

import config
from aimbot import capture, hud, targeting
from aimbot.controls import Mouse, is_key_held
from aimbot.detector import Detector


def main():
	capture.enable_dpi_awareness()

	hwnd = capture.find_window(config.WINDOW_TITLE)
	if hwnd is None:
		print(f"[ERROR] Window '{config.WINDOW_TITLE}' not found. Open GTA V first.")
		sys.exit(1)
	capture.place_window(hwnd, config.INITIAL_WINDOW_X, config.INITIAL_WINDOW_Y,
		config.INITIAL_WINDOW_WIDTH, config.INITIAL_WINDOW_HEIGHT)

	detector = Detector(config.MODEL_CFG, config.MODEL_WEIGHTS, config.CLASSES_FILE,
		config.USE_CUDA, config.BLOB_SIZE, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
	mouse = Mouse()

	print("[INFO] Aimbot running. Hold '{}' to aim. Press 'Q' on preview to quit.".format(
		config.AIM_ACTIVATION_KEY))

	# The capture reads from the screen, so the preview must not sit on top of
	# the game (it would capture itself). Put it just right of the game window.
	cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
	cv2.moveWindow(config.WINDOW_NAME, capture.window_right_edge(hwnd) + 10, 0)

	last_time = time.time()
	fps_smooth = 0.0

	try:
		while True:
			frame = capture.grab_frame(hwnd)
			if frame is None:
				time.sleep(0.01)
				continue

			image = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
			frame_h, frame_w = image.shape[:2]

			# FPS with smoothing
			now = time.time()
			dt = now - last_time
			fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt if dt > 0 else 0)
			last_time = now

			detections = detector.detect(image)
			people = [d for d in detections if d.label == config.TARGET_CLASS]
			own = targeting.find_self(people, frame_w, frame_h)
			enemies = [d for d in people if d is not own]

			for det in detections:
				is_person = det.label == config.TARGET_CLASS
				if det is own:
					color, label = hud.SELF_COLOR, "self (ignored)"
				else:
					color = detector.color_for(det)
					label = "{}: {:.2f}".format(det.label, det.confidence)
				hud.draw_detection(image, det, color, label,
					show_geometry=config.SHOW_SELF_DEBUG and is_person)

			aim_active = is_key_held(config.AIM_ACTIVATION_KEY)
			should_fire = False

			target = targeting.closest_to_crosshair(enemies, frame_w, frame_h)
			if target is not None:
				hud.draw_target_marker(image, target)

				if aim_active:
					# Move towards the target, damped for less jerky movement
					dx = int((target.cx - 0.5 * frame_w) * config.AIM_SMOOTHING)
					dy = int((target.cy - 0.5 * frame_h) * config.AIM_SMOOTHING)
					mouse.move(dx, dy)

					if config.AUTO_FIRE:
						should_fire = targeting.crosshair_inside(target, frame_w, frame_h)

			# Press/release only on state change; releases when target lost or key let go
			mouse.set_firing(should_fire)

			hud.draw_self_areas(image)
			hud.draw_status(image, fps_smooth, len(detections), len(enemies), aim_active)

			cv2.imshow(config.WINDOW_NAME, image)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		# Never leave the left button stuck down (quit, Ctrl+C or crash)
		mouse.set_firing(False)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
