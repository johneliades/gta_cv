"""Drawing the preview overlay."""

import cv2

import config

SELF_COLOR = (128, 128, 128)


def draw_detection(img, det, color, label, show_geometry):
	frame_h, frame_w = img.shape[:2]
	cv2.rectangle(img, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
	cv2.putText(img, label, (det.x, det.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# Tuning aid: height and bottom-centre (x, bottom) as screen fractions
	if show_geometry:
		dbg = "h={:.2f} x={:.2f} b={:.2f}".format(
			det.h / float(frame_h),
			(det.x + det.w / 2.0) / frame_w,
			(det.y + det.h) / float(frame_h))
		cv2.putText(img, dbg, (det.x, min(det.y + det.h + 15, frame_h - 5)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_target_marker(img, det):
	cv2.circle(img, (det.cx, det.cy), 5, (0, 0, 255), -1)


def draw_self_areas(img):
	"""A box whose bottom-centre lands in one of these (and is tall enough)
	counts as self. Tune SELF_* in config.py."""
	if not config.SELF_EXCLUSION_ENABLED:
		return
	frame_h, frame_w = img.shape[:2]
	# Third person (bottom-left), yellow
	cv2.rectangle(img,
		(0, int(config.SELF_THIRD_PERSON_MIN_BOTTOM * frame_h)),
		(int(config.SELF_THIRD_PERSON_MAX_X * frame_w), frame_h - 1),
		(0, 255, 255), 1)
	# First person (bottom middle), magenta
	cv2.rectangle(img,
		(int((0.5 - config.SELF_FIRST_PERSON_HALF_WIDTH) * frame_w),
			int(config.SELF_FIRST_PERSON_MIN_BOTTOM * frame_h)),
		(int((0.5 + config.SELF_FIRST_PERSON_HALF_WIDTH) * frame_w), frame_h - 1),
		(255, 0, 255), 1)


def draw_status(img, fps, detection_count, target_count, aim_active):
	frame_h, frame_w = img.shape[:2]

	# Crosshair
	cv2.drawMarker(img, (frame_w // 2, frame_h // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)

	if config.SHOW_FPS:
		cv2.putText(img, "FPS: {:.0f}".format(fps), (10, 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	if config.SHOW_DETECTION_COUNT:
		cv2.putText(img, "Detections: {}  |  Targets: {}".format(detection_count, target_count),
			(10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

	status_color = (0, 255, 0) if aim_active else (0, 0, 255)
	status_text = "AIM: ON" if aim_active else "AIM: OFF"
	cv2.putText(img, status_text, (frame_w - 130, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
