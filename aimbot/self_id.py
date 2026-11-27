"""Finding the player's own character among the person boxes.

Each person box gets a score from three signals, and only the best box above
a threshold counts as self (there is exactly one player):

  motion      In third person the camera follows you, so when it turns the
              background slides across the screen while your character stays
              put. The background shift is measured with phase correlation on
              a small greyscale copy of the frame; each box is followed from
              frame to frame by overlap, and a box that stays fixed while the
              background moves scores high. Remembered per box, so the last
              known self keeps its score while the camera is still.
  appearance  How closely the box's colours (HSV histogram) match a saved
              profiles of your character (one per lighting it was seen in;
              the best match counts). Brightness is evened out first, so a
              darker or brighter scene alone doesn't break the match. A profile
              is added when you pick yourself (below), and refreshed on its own
              from frames where the motion signal is sure.
  position    The fixed screen-area rule (targeting.in_self_area), as a weak hint.

A signal with no evidence yet (no profile saved, a box not yet seen while the
camera moved) counts as 0.5, "don't know", so it neither helps nor hurts. With
no evidence at all only the position rule is left, so the old area rule is the
fallback until the camera moves or a profile is saved.

Manual pick: SELF_SELECT_NEXT_KEY starts selection mode, and each further
press moves the highlight to the next person on screen (left to right), then
to a Cancel placeholder, then round again. SELF_SELECT_CONFIRM_KEY picks the
highlighted person, or leaves selection mode without a pick on Cancel. The picked box is self for as long
as it stays tracked, whatever the scores say, and its colours are saved as a
profile, so it is still recognised after it is lost.
"""

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

import config
from aimbot import targeting

_UNKNOWN = 0.5
_CANCEL = object()  # the Cancel placeholder in the selection cycle

# HSV histogram bins (hue, saturation, value). Coarse value bins keep black and
# white clothing (no reliable hue) apart without being thrown by lighting.
_HIST_BINS = [16, 8, 8]
_HIST_RANGES = [0, 180, 0, 256, 0, 256]
_TARGET_BRIGHTNESS = 110.0  # every crop is scaled to this mean grey level first


@dataclass
class SelfScore:
	motion: float      # 0..1, 0.5 = unknown
	appearance: float  # 0..1, 0.5 = unknown
	position: float    # 0 or 1
	total: float
	similarity: float = None  # raw histogram similarity to the best profile (0..1), None without one


class _Track:
	"""One person box followed across frames."""

	def __init__(self, box):
		self.box = box        # (x, y, w, h), pixels
		self.misses = 0       # frames in a row without a matching box
		self.still = _UNKNOWN # running average of "stayed fixed while the camera moved"
		self.samples = 0      # frames that average is based on

	@property
	def motion_score(self):
		# Pulled towards "unknown" until there are enough samples to trust it
		trust = min(1.0, self.samples / float(max(1, config.SELF_MOTION_MIN_SAMPLES)))
		return _UNKNOWN + (self.still - _UNKNOWN) * trust


def _iou(a, b):
	ax, ay, aw, ah = a
	bx, by, bw, bh = b
	iw = min(ax + aw, bx + bw) - max(ax, bx)
	ih = min(ay + ah, by + bh) - max(ay, by)
	if iw <= 0 or ih <= 0:
		return 0.0
	inter = iw * ih
	return inter / float(aw * ah + bw * bh - inter)


def _centre(box):
	x, y, w, h = box
	return x + w / 2.0, y + h / 2.0


def _box(det):
	return det.x, det.y, det.w, det.h


def _histogram(image, det):
	"""Normalised HSV histogram of the middle of the box (its edges are mostly
	background), or None if the box is off-screen or too small. The crop is
	first scaled to a fixed average brightness, so the same outfit in a darker
	or brighter scene gives about the same histogram."""
	frame_h, frame_w = image.shape[:2]
	mx, my = det.w * 0.2, det.h * 0.1
	x0, y0 = max(0, int(det.x + mx)), max(0, int(det.y + my))
	x1, y1 = min(frame_w, int(det.x + det.w - mx)), min(frame_h, int(det.y + det.h - my))
	if x1 - x0 < 4 or y1 - y0 < 4:
		return None
	crop = image[y0:y1, x0:x1]
	mean = cv2.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))[0]
	gain = min(4.0, max(0.25, _TARGET_BRIGHTNESS / max(mean, 1.0)))
	hsv = cv2.cvtColor(cv2.convertScaleAbs(crop, alpha=gain), cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_BINS, _HIST_RANGES)
	cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
	return hist


class SelfIdentifier:
	def __init__(self):
		self.profiles = self._load_profiles()  # colour profiles, oldest first
		self._profile_dirty = False
		self.camera_shift = None  # (dx, dy) background shift this frame in pixels, or None
		self._prev_small = None
		self._window = None
		self._tracks = []
		self._self_track = None
		self._pinned = None       # track picked by hand: self while it lives
		self.selecting = False    # selection mode on
		self._candidate = None    # highlighted track (or _CANCEL) in selection mode
		self._frame_people = []   # this frame's person boxes...
		self._frame_tracks = []   # ...and their tracks

	# ─── Per frame ────────────────────────────────────────────────────────

	def update(self, image, people):
		"""Score every person box. Returns (self detection or None, list of
		SelfScore aligned with `people`)."""
		frame_h, frame_w = image.shape[:2]
		self.camera_shift = self._measure_camera_shift(image)
		moving = (self.camera_shift is not None
			and math.hypot(*self.camera_shift) >= config.SELF_MOTION_MIN_SHIFT * frame_w)
		tracks = self._update_tracks(people, self.camera_shift if moving else None)
		self._frame_people, self._frame_tracks = list(people), tracks
		if not config.SELF_EXCLUSION_ENABLED:  # boxes are still tracked, for the target lock
			return None, [None] * len(people)
		if self.selecting:
			self._ensure_candidate()

		scores = []
		for det, track in zip(people, tracks):
			hist = _histogram(image, det) if self.profiles else None
			similarity = None
			appearance = _UNKNOWN
			if hist is not None:
				similarity, _ = self._best_match(hist)
				span = max(1e-6, config.SELF_APPEARANCE_MATCH - config.SELF_APPEARANCE_NO_MATCH)
				appearance = min(1.0, max(0.0, (similarity - config.SELF_APPEARANCE_NO_MATCH) / span))
			position = 1.0 if targeting.in_self_area(det, frame_w, frame_h) else 0.0
			motion = track.motion_score
			total = (config.SELF_WEIGHT_MOTION * motion
				+ config.SELF_WEIGHT_APPEARANCE * appearance
				+ config.SELF_WEIGHT_POSITION * position)
			if track is self._self_track:
				total += config.SELF_STICKY_BONUS  # avoid flicker between close scores
			scores.append(SelfScore(motion, appearance, position, total, similarity))

		if self._pinned in tracks:
			best = tracks.index(self._pinned)  # picked by hand: overrides the scores
		else:
			best = max(range(len(people)), key=lambda i: scores[i].total, default=None)
		if best is None or (tracks[best] is not self._pinned
				and scores[best].total < config.SELF_SCORE_THRESHOLD):
			self._self_track = None
			return None, scores

		own = people[best]
		self._self_track = tracks[best]
		if moving and config.SELF_APPEARANCE_AUTO_UPDATE:
			self._auto_update_profile(image, own, tracks[best])
		return own, scores

	def reset(self):
		"""Forget frame-to-frame state (no frame, or the target window changed)."""
		self._prev_small = None
		self.camera_shift = None
		self._tracks = []
		self._self_track = None
		self._pinned = None
		self.selecting = False
		self._candidate = None
		self._frame_people, self._frame_tracks = [], []

	# ─── Box tracking (also used by the target lock) ─────────────────────

	def track_of(self, det):
		"""The track of one of this frame's person boxes (None if unknown)."""
		for person, track in zip(self._frame_people, self._frame_tracks):
			if person is det:
				return track
		return None

	def is_tracked(self, track):
		"""True while this person is remembered, i.e. missing for at most
		TRACK_MAX_MISSES frames in a row."""
		return track is not None and track in self._tracks

	def on_screen(self, track):
		"""True if this person has a box in this frame."""
		return track is not None and track in self._frame_tracks

	@property
	def pinned(self):
		"""True while a box picked by hand is being tracked."""
		return self._pinned is not None

	# ─── Signal 1: stays put while the camera turns ──────────────────────

	def _measure_camera_shift(self, image):
		"""How far the background moved since the last frame, (dx, dy) in
		full-resolution pixels, or None if unknown or unreliable."""
		frame_h, frame_w = image.shape[:2]
		scale = config.SELF_MOTION_WIDTH / float(frame_w)
		size = (config.SELF_MOTION_WIDTH, max(8, int(round(frame_h * scale))))
		small = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA),
			cv2.COLOR_BGR2GRAY).astype(np.float32)

		prev, self._prev_small = self._prev_small, small
		if prev is None or prev.shape != small.shape:
			self._tracks = []  # first frame, or the frame size changed
			return None
		if self._window is None or self._window.shape != small.shape:
			self._window = cv2.createHanningWindow(size, cv2.CV_32F)

		(dx, dy), response = cv2.phaseCorrelate(prev, small, self._window)
		if response < config.SELF_MOTION_MIN_RESPONSE:
			return None  # no clear single shift (e.g. featureless or cut to a new scene)
		return dx / scale, dy / scale

	def _update_tracks(self, people, shift):
		"""Match each box to a track from earlier frames (by overlap, also
		trying the track moved along with the background), update the tracks'
		stillness while the camera moves, and return the track of each box."""
		boxes = [_box(d) for d in people]
		sx, sy = shift if shift is not None else (0.0, 0.0)

		pairs = []
		for ti, track in enumerate(self._tracks):
			x, y, w, h = track.box
			moved = (x + sx, y + sy, w, h)
			for bi, box in enumerate(boxes):
				overlap = max(_iou(track.box, box), _iou(moved, box))
				if overlap >= config.SELF_TRACK_MIN_IOU:
					pairs.append((overlap, ti, bi))
		pairs.sort(reverse=True)

		matched = [None] * len(boxes)
		used = set()
		for _, ti, bi in pairs:
			if ti in used or matched[bi] is not None:
				continue
			used.add(ti)
			track = self._tracks[ti]
			if shift is not None:
				# A world-fixed box moves with the background (d ~ s); our
				# character stays on screen (d ~ 0). 1 = screen-fixed.
				(px, py), (cx, cy) = _centre(track.box), _centre(boxes[bi])
				dx, dy = cx - px, cy - py
				to_world = math.hypot(dx - sx, dy - sy)
				to_screen = math.hypot(dx, dy)
				stillness = to_world / (to_world + to_screen + 1e-6)
				rate = config.SELF_MOTION_SMOOTHING
				track.still = (1.0 - rate) * track.still + rate * stillness
				track.samples += 1
			track.box = boxes[bi]
			track.misses = 0
			matched[bi] = track

		for ti, track in enumerate(self._tracks):
			if ti not in used:
				track.misses += 1
		self._tracks = [t for t in self._tracks if t.misses <= config.TRACK_MAX_MISSES]
		for bi, box in enumerate(boxes):
			if matched[bi] is None:
				matched[bi] = _Track(box)
				self._tracks.append(matched[bi])
		if self._self_track not in self._tracks:
			self._self_track = None
		if self._pinned is not None and self._pinned not in self._tracks:
			self._pinned = None
			print("[INFO] Lost the character you picked; back to automatic detection.")
		return matched

	# ─── Signal 2: looks like you ────────────────────────────────────────

	def _best_match(self, hist):
		"""(similarity 0..1, index) of the saved profile closest to `hist`."""
		sims = [1.0 - cv2.compareHist(p, hist, cv2.HISTCMP_BHATTACHARYYA) for p in self.profiles]
		best = int(np.argmax(sims))
		return sims[best], best

	def _add_profile(self, image, det):
		"""Save this box's colours as a profile. A saved one at least
		SELF_PROFILE_SAME similar is replaced (same lighting, newer look);
		otherwise it is added, dropping the oldest beyond SELF_PROFILE_SLOTS.
		Returns "added" / "replaced", or None if the box can't be measured."""
		hist = _histogram(image, det) if det is not None else None
		if hist is None:
			return None
		result = "added"
		if self.profiles:
			similarity, index = self._best_match(hist)
			if similarity >= config.SELF_PROFILE_SAME:
				del self.profiles[index]
				result = "replaced"
		self.profiles.append(hist)
		del self.profiles[:-max(1, config.SELF_PROFILE_SLOTS)]
		self._profile_dirty = True
		self._write_profiles()
		return result

	def _auto_update_profile(self, image, own, track):
		"""Blend the self box into its closest profile while the motion signal
		alone is sure it is self (so slow changes are followed)."""
		if track.samples < config.SELF_MOTION_MIN_SAMPLES or track.still < config.SELF_AUTO_UPDATE_MIN_MOTION:
			return
		hist = _histogram(image, own)
		if hist is None:
			return
		if not self.profiles:
			self.profiles.append(hist)
			print("[INFO] Learned your character's colour profile from camera motion.")
		else:
			_, index = self._best_match(hist)
			rate = config.SELF_AUTO_UPDATE_RATE
			self.profiles[index] = cv2.addWeighted(self.profiles[index], 1.0 - rate, hist, rate, 0.0)
		self._profile_dirty = True

	# ─── Manual pick ─────────────────────────────────────────────────────

	@property
	def candidate(self):
		"""The highlighted person box in selection mode, or None."""
		for det, track in zip(self._frame_people, self._frame_tracks):
			if track is self._candidate:
				return det
		return None

	def _ordered_tracks(self):
		"""Tracks of this frame's boxes, left to right."""
		pairs = sorted(zip(self._frame_people, self._frame_tracks), key=lambda p: p[0].cx)
		return [track for _, track in pairs]

	def _ensure_candidate(self):
		"""Keep the highlight on its person while they are tracked, even
		through a few frames without a box (no jumping); once they are gone,
		move it to the first person from the left, or to Cancel."""
		if self._candidate is _CANCEL or self.is_tracked(self._candidate):
			return
		on_screen = self._ordered_tracks()
		self._candidate = on_screen[0] if on_screen else _CANCEL

	@property
	def cancel_highlighted(self):
		"""True while the Cancel placeholder is highlighted in selection mode."""
		return self.selecting and self._candidate is _CANCEL

	def select_next(self):
		"""First press: start selection mode, highlighting the current self (or
		the first person from the left). Further presses: move the highlight to
		the next person, left to right, then to Cancel, then round again."""
		if not self.selecting:
			self.selecting = True
			self._candidate = self._self_track
			self._ensure_candidate()
			print("[INFO] Selection mode: '{}' next, '{}' pick (pick Cancel to leave).".format(
				config.SELF_SELECT_NEXT_KEY, config.SELF_SELECT_CONFIRM_KEY))
			return
		cycle = self._ordered_tracks() + [_CANCEL]
		i = cycle.index(self._candidate) + 1 if self._candidate in cycle else 0
		self._candidate = cycle[i % len(cycle)]

	def confirm_selection(self, image):
		"""Pick the highlighted box as self: pinned while tracked, and its
		colours saved as a profile. Returns the picked box, or None."""
		if not self.selecting:
			return None
		if self._candidate is _CANCEL:
			self.selecting = False
			self._candidate = None
			print("[INFO] Selection cancelled.")
			return None
		det = self.candidate
		if det is None:
			print("[WARN] No person on screen to pick; still in selection mode.")
			return None
		track = self._candidate
		self._pinned = self._self_track = track
		# Its motion history now counts as certain, so automatic detection
		# keeps choosing it if the pin is lost while the box is still around
		track.still = 1.0
		track.samples = max(track.samples, config.SELF_MOTION_MIN_SAMPLES)
		saved = self._add_profile(image, det)
		self.selecting = False
		self._candidate = None
		print("[INFO] Picked your character (box at x={}, y={}, {}x{}){}.".format(
			det.x, det.y, det.w, det.h,
			"; colour profile {} ({} kept)".format(saved, len(self.profiles)) if saved else ""))
		return det

	def clear_profiles(self):
		"""Forget all colour profiles, also on disk (e.g. after switching to
		another character). Pick yourself again afterwards."""
		self.profiles = []
		self._profile_dirty = False
		path = config.SELF_PROFILE_FILE
		if path and os.path.isfile(path):
			try:
				os.remove(path)
			except OSError as e:
				print("[WARN] Could not delete '{}': {}".format(path, e))
		print("[INFO] Cleared your character's colour profiles. Press '{}' to pick yourself.".format(
			config.SELF_SELECT_NEXT_KEY))

	def close(self):
		"""Write the profiles to disk if they changed since last written."""
		if self._profile_dirty:
			self._write_profiles()

	def _write_profiles(self):
		if not config.SELF_PROFILE_FILE or not self.profiles:
			return
		try:
			np.save(config.SELF_PROFILE_FILE, np.stack(self.profiles))
			self._profile_dirty = False
		except OSError as e:
			print("[WARN] Could not save the colour profiles to '{}': {}".format(config.SELF_PROFILE_FILE, e))

	@staticmethod
	def _load_profiles():
		path = config.SELF_PROFILE_FILE
		if not path or not os.path.isfile(path):
			return []
		try:
			stack = np.load(path).astype(np.float32)
		except (OSError, ValueError) as e:
			print("[WARN] Could not load the colour profiles '{}': {}".format(path, e))
			return []
		if stack.ndim != len(_HIST_BINS) + 1 or list(stack.shape[1:]) != _HIST_BINS:
			print("[WARN] Ignoring colour profile '{}' (saved by an older version); "
				"pick yourself again.".format(path))
			return []
		print("[INFO] Loaded {} colour profile(s) of your character from '{}'.".format(len(stack), path))
		return list(stack[-max(1, config.SELF_PROFILE_SLOTS):])
