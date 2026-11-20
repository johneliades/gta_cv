"""Target selection: ignoring the player's own character and picking the target."""

import config


def is_own_character(det, frame_w, frame_h):
	"""Heuristic: could this 'person' box be the player's own character/arms?

	Judged by box height and its bottom-centre point (middle of the bottom
	edge), as fractions of the frame. No overlap test.
	"""
	if not config.SELF_EXCLUSION_ENABLED or det.w <= 0 or det.h <= 0:
		return False

	h = det.h / float(frame_h)
	bottom = (det.y + det.h) / float(frame_h)
	center_x = (det.x + det.w / 2.0) / float(frame_w)

	# Third person: large character, bottom-left of the crosshair
	third_person = (h >= config.SELF_THIRD_PERSON_MIN_HEIGHT
		and bottom >= config.SELF_THIRD_PERSON_MIN_BOTTOM
		and center_x <= config.SELF_THIRD_PERSON_MAX_X)

	# First person: arms + gun at the bottom middle (smaller when aiming)
	first_person = (h >= config.SELF_FIRST_PERSON_MIN_HEIGHT
		and bottom >= config.SELF_FIRST_PERSON_MIN_BOTTOM
		and abs(center_x - 0.5) <= config.SELF_FIRST_PERSON_HALF_WIDTH)

	return third_person or first_person


def find_self(people, frame_w, frame_h):
	"""There is exactly one player, so at most ONE box is self: among the
	person boxes matching the self rules, the largest one (the own character
	is the closest person to the camera). Returns None if none match."""
	candidates = [d for d in people if is_own_character(d, frame_w, frame_h)]
	return max(candidates, key=lambda d: d.area) if candidates else None


def closest_to_crosshair(people, frame_w, frame_h):
	"""The person whose centre is nearest the screen centre, or None."""
	if not people:
		return None
	return min(people, key=lambda d: (0.5 * frame_w - d.cx) ** 2 + (0.5 * frame_h - d.cy) ** 2)


def crosshair_inside(det, frame_w, frame_h):
	"""True if the screen centre (crosshair) lies inside the detection box."""
	return (det.x < 0.5 * frame_w < det.x + det.w
		and det.y < 0.5 * frame_h < det.y + det.h)
