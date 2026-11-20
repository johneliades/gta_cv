"""GTA V computer-vision aim assist package.

This runs before any submodule (and therefore before cv2 is imported):
a CUDA-enabled OpenCV build needs the CUDA/cuDNN DLLs (cublas, cufft, npp,
cudnn). Python 3.8+ ignores PATH when loading extension DLLs, so the CUDA bin
folder is registered explicitly.
"""

import os

_cuda_path = os.environ.get("CUDA_PATH")
if _cuda_path and hasattr(os, "add_dll_directory"):
	_cuda_bin = os.path.join(_cuda_path, "bin")
	if os.path.isdir(_cuda_bin):
		os.add_dll_directory(_cuda_bin)
