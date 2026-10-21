import sys
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

print("Python:", sys.version)
print("Interpreter:", sys.executable)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("Pose Landmarker:", vision.PoseLandmarker)