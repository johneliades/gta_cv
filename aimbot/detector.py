"""YOLO model loading and object detection (OpenCV DNN)."""

import sys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
	class_id: int
	label: str
	confidence: float
	x: int  # top-left corner, pixels
	y: int
	w: int
	h: int
	cx: int  # centre, pixels
	cy: int


def cuda_available():
	"""True only if this OpenCV build has CUDA support and a CUDA device is present.
	The pip 'opencv-python' wheels are CPU-only, so this is False for them."""
	try:
		return cv2.cuda.getCudaEnabledDeviceCount() > 0
	except Exception:
		return False


class Detector:
	def __init__(self, cfg_path, weights_path, classes_path, use_cuda,
			blob_size, confidence_threshold, nms_threshold):
		self.blob_size = blob_size
		self.confidence_threshold = confidence_threshold
		self.nms_threshold = nms_threshold

		try:
			with open(classes_path) as f:
				self.labels = f.read().strip().split("\n")
		except FileNotFoundError:
			print(f"[ERROR] Classes file '{classes_path}' not found.")
			sys.exit(1)

		# Stable per-class colours for drawing
		rng = np.random.RandomState(42)
		self.colors = rng.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

		if not hasattr(cv2.dnn, "readNetFromDarknet"):
			print(f"[ERROR] OpenCV {cv2.__version__} has no Darknet loader (removed in OpenCV 5).")
			print("        Install OpenCV 4.x:  pip install \"opencv-python>=4.5.0,<5\"")
			sys.exit(1)

		try:
			self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
		except cv2.error:
			print(f"[ERROR] Failed to load YOLO model. Ensure '{weights_path}' exists.")
			print("        Download from: https://github.com/AlexeyAB/darknet/releases")
			sys.exit(1)

		if use_cuda and cuda_available():
			print("[INFO] Using CUDA backend.")
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		else:
			if use_cuda:
				print("[WARN] USE_CUDA is enabled but this OpenCV build has no CUDA support "
					"(pip wheels are CPU-only). Falling back to CPU.")
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
			self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

		layer_names = self.net.getLayerNames()
		self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

	def color_for(self, det):
		return [int(c) for c in self.colors[det.class_id]]

	def detect(self, image):
		"""Run the model on a BGR image; return detections after NMS."""
		frame_h, frame_w = image.shape[:2]

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self.blob_size, swapRB=True, crop=False)
		self.net.setInput(blob)
		outputs = self.net.forward(self.output_layers)

		boxes, confidences, class_ids = [], [], []
		scale = np.array([frame_w, frame_h, frame_w, frame_h])
		for output in outputs:
			for row in output:
				scores = row[5:]
				class_id = int(np.argmax(scores))
				confidence = float(scores[class_id])
				if confidence > self.confidence_threshold:
					cx, cy, w, h = (row[0:4] * scale).astype("int")
					boxes.append([int(cx), int(cy), int(w), int(h)])
					confidences.append(confidence)
					class_ids.append(class_id)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

		detections = []
		for i in np.array(idxs).flatten():
			cx, cy, w, h = boxes[i]
			detections.append(Detection(
				class_id=class_ids[i],
				label=self.labels[class_ids[i]],
				confidence=confidences[i],
				x=int(cx - w / 2), y=int(cy - h / 2), w=w, h=h,
				cx=cx, cy=cy))
		return detections
