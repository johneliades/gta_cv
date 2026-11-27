"""Drawing the HUD on the overlay canvas (see overlay.py).

The canvas is BGRA and starts fully transparent; everything here is drawn
opaque (alpha 255), with a thin dark outline so it stays readable on bright
parts of the game. Colours are passed in as plain BGR.
"""

import cv2
import numpy as np

import config

SELECT_COLOR = (255, 255, 0)  # cyan: highlighted person in selection mode
OUTLINE = (0, 0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bgra(color):
	return (int(color[0]), int(color[1]), int(color[2]), 255)


def _text(img, text, org, scale, color, thickness):
	cv2.putText(img, text, org, FONT, scale, OUTLINE, thickness + 2, cv2.LINE_AA)
	cv2.putText(img, text, org, FONT, scale, _bgra(color), thickness, cv2.LINE_AA)


_ink_cache = {}


def _ink_offsets(text, scale, thickness):
	"""(right, top) of the pixels _text() actually draws, relative to its
	origin (left end of the baseline), outline included. getTextSize includes
	blank space beside the letters, so it is measured by drawing once."""
	key = (text, scale, thickness)
	if key not in _ink_cache:
		(w, h), base = cv2.getTextSize(text, FONT, scale, thickness + 2)
		pad = 10
		mask = np.zeros((h + base + 2 * pad, w + 2 * pad), np.uint8)
		cv2.putText(mask, text, (pad, pad + h), FONT, scale, 255, thickness + 2, cv2.LINE_AA)
		ys, xs = np.nonzero(mask)
		_ink_cache[key] = (xs.max() - pad, ys.min() - (pad + h))
	return _ink_cache[key]


def _rect(img, pt1, pt2, color, thickness):
	cv2.rectangle(img, pt1, pt2, OUTLINE, thickness + 2)
	cv2.rectangle(img, pt1, pt2, _bgra(color), thickness)


def draw_detection(img, det, color, label, show_geometry, score=None):
	frame_h, frame_w = img.shape[:2]
	_rect(img, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
	_text(img, label, (det.x, det.y - 5), 0.5, color, 2)

	# Tuning aid: height and bottom-centre (x, bottom) as screen fractions
	if show_geometry:
		dbg = "h={:.2f} x={:.2f} b={:.2f}".format(
			det.h / float(frame_h),
			(det.x + det.w / 2.0) / frame_w,
			(det.y + det.h) / float(frame_h))
		_text(img, dbg, (det.x, min(det.y + det.h + 15, frame_h - 5)), 0.45, color, 1)
		# Self score: motion, appearance (raw colour similarity), position = total
		if score is not None:
			sim = "" if score.similarity is None else " ({:.2f})".format(score.similarity)
			dbg = "m={:.2f} a={:.2f}{} p={:.0f} = {:.2f}".format(
				score.motion, score.appearance, sim, score.position, score.total)
			_text(img, dbg, (det.x, min(det.y + det.h + 32, frame_h - 5)), 0.45, color, 1)


def draw_selection(img, det, cancel_highlighted):
	"""Selection mode: a thick highlight on the person that would be picked,
	a Cancel placeholder (highlighted when it is the current choice), and the
	keys to use."""
	frame_h, frame_w = img.shape[:2]
	if det is not None:
		pad = 4
		_rect(img, (det.x - pad, det.y - pad), (det.x + det.w + pad, det.y + det.h + pad),
			SELECT_COLOR, 3)
		_text(img, "this is me?", (det.x, max(det.y - 25, 15)), 0.6, SELECT_COLOR, 2)

	# Cancel placeholder, bottom middle
	label = "CANCEL"
	(tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
	left, bottom = (frame_w - tw) // 2 - 12, frame_h - 45
	box = ((left, bottom - th - 16), (left + tw + 24, bottom))
	color = SELECT_COLOR if cancel_highlighted else (160, 160, 160)
	_rect(img, box[0], box[1], color, 3 if cancel_highlighted else 1)
	_text(img, label, (left + 12, bottom - 8), 0.7, color, 2)

	hint = "SELECT YOURSELF: {} next | {} pick".format(
		config.SELF_SELECT_NEXT_KEY, config.SELF_SELECT_CONFIRM_KEY)
	_text(img, hint, (10, frame_h - 15), 0.6, SELECT_COLOR, 2)


def draw_target_marker(img, zone):
	"""The target's aim area (head or body centre: the auto-fire zone) and a
	red dot on its centre (aim point)."""
	left, top, right, bottom = zone
	_rect(img, (left, top), (right, bottom), (0, 0, 255), 1)
	centre = ((left + right) // 2, (top + bottom) // 2)
	cv2.circle(img, centre, 4, OUTLINE, -1, cv2.LINE_AA)
	cv2.circle(img, centre, 3, _bgra((0, 0, 255)), -1, cv2.LINE_AA)


def draw_status(img, fps, detection_count, target_count, aim_active, aim_point, self_status=None):
	frame_h, frame_w = img.shape[:2]

	# Crosshair
	centre = (frame_w // 2, frame_h // 2)
	cv2.drawMarker(img, centre, OUTLINE, cv2.MARKER_CROSS, 20, 3)
	cv2.drawMarker(img, centre, _bgra((0, 255, 0)), cv2.MARKER_CROSS, 20, 1)

	if config.SHOW_FPS:
		_text(img, "FPS: {:.0f}".format(fps), (10, 25), 0.7, (0, 255, 0), 2)

	if config.SHOW_DETECTION_COUNT:
		_text(img, "Detections: {}  |  Targets: {}".format(detection_count, target_count),
			(10, 55), 0.6, (0, 255, 255), 2)

	if self_status:
		_text(img, self_status, (10, 85), 0.55, (200, 200, 200), 1)

	status_color = (0, 255, 0) if aim_active else (0, 0, 255)
	status_text = "AIM: {} | {}".format("ON" if aim_active else "OFF",
		"CENTRE" if aim_point == "centre" else "HEAD")
	# Top-right corner, the same gap from the top and the right edge
	margin = 10
	right, top = _ink_offsets(status_text, 0.7, 2)
	_text(img, status_text, (frame_w - 1 - margin - right, margin - top), 0.7, status_color, 2)
