"""Target selection: the self-area position rule and picking the target."""

import config


def in_self_area(det, frame_w, frame_h):
	"""Position rule: is this 'person' box where the player's own
	character/arms usually are? Only a weak hint, one of the three signals
	scored in self_id.py.

	Judged by box height and its bottom-centre point (middle of the bottom
	edge), as fractions of the frame. No overlap test.
	"""
	if det.w <= 0 or det.h <= 0:
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


HEAD_MIN_SIZE = 6  # px: an estimated head area is never smaller than this


def head_box(person, heads):
	"""(left, top, right, bottom) of this person's head. Aim at its centre;
	auto-fire only while the crosshair is inside it.

	A detected head is used if one lies in the upper half of the person's box
	(only these are searched, so a head is matched to its own body); the most
	confident wins. Otherwise - always, with a COCO model, which has no head
	class - it is estimated from the box: a square HEAD_SIZE_FRACTION of the
	box height, centred horizontally, its centre HEAD_Y_FRACTION of the box
	height below the top.
	"""
	upper_bottom = person.y + person.h / 2.0
	inside = [h for h in heads
		if person.x <= h.cx <= person.x + person.w and person.y <= h.cy <= upper_bottom]
	if inside:
		head = max(inside, key=lambda h: h.confidence)
		return head.x, head.y, head.x + head.w, head.y + head.h
	half = max(HEAD_MIN_SIZE, min(person.h * config.HEAD_SIZE_FRACTION, person.w)) / 2.0
	cx, cy = person.cx, person.y + person.h * config.HEAD_Y_FRACTION
	return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def box_centre(box):
	left, top, right, bottom = box
	return (left + right) // 2, (top + bottom) // 2


def pick_target(people, heads, frame_w, frame_h):
	"""(person, head box) for the person whose head is nearest the crosshair,
	or None if there are no people."""
	if not people:
		return None
	targets = [(person, head_box(person, heads)) for person in people]

	def distance(target):
		x, y = box_centre(target[1])
		return (0.5 * frame_w - x) ** 2 + (0.5 * frame_h - y) ** 2
	return min(targets, key=distance)


class TargetLock:
	"""While the aim key is held, keep aiming at the same person: the one
	closest to the crosshair when aiming started. If their box disappears for
	a few frames (a detection gap), wait for them to come back instead of
	jumping to someone else; only once they are forgotten (TRACK_MAX_MISSES
	frames) is the next closest person picked. Without the aim key nothing is
	locked, so the marker always shows who would be picked."""

	def __init__(self):
		self._track = None

	def pick(self, enemies, heads, frame_w, frame_h, tracker, aim_active):
		"""(person, head box) to aim at, or None (nobody, or waiting for the
		locked person to reappear). `tracker` is the SelfIdentifier, which
		follows every person box across frames."""
		if not aim_active:
			self._track = None
		elif self._track is not None and tracker.is_tracked(self._track):
			for det in enemies:
				if tracker.track_of(det) is self._track:
					return det, head_box(det, heads)
			if not tracker.on_screen(self._track):
				return None  # not detected this frame: wait for them
			# on screen but no longer an enemy (now counted as self): re-pick

		picked = pick_target(enemies, heads, frame_w, frame_h)
		self._track = tracker.track_of(picked[0]) if picked and aim_active else None
		return picked


def crosshair_inside(box, frame_w, frame_h):
	"""True if the screen centre (crosshair) lies inside (left, top, right, bottom)."""
	left, top, right, bottom = box
	return left <= 0.5 * frame_w <= right and top <= 0.5 * frame_h <= bottom
