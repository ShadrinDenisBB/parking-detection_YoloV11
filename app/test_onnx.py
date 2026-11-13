import onnxruntime as ort
import numpy as np
import cv2
import time

# Load ONNX model
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, 1280, 1280]

# Load test image
img = cv2.imread("test1 (1).jpg")
img = cv2.resize(img, (1280, 1280))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC to CHW
img = np.expand_dims(img, axis=0)   # Add batch

# Warm-up
for _ in range(5):
    session.run(None, {input_name: img})

# Benchmark
times = []
for _ in range(20):
    start = time.time()
    outputs = session.run(None, {input_name: img})
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000  # ms
fps = 1000 / avg_time

print(f"Average inference: {avg_time:.1f} ms")
print(f"FPS: {fps:.1f}")