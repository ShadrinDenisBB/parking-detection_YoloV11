from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Export to ONNX
# imgsz=1280 — same as training
# simplify=True — optimize graph
# opset=12 — compatibility
model.export(
    format="onnx",
    imgsz=1280,
    simplify=True,
    opset=12,
    half=False,      # True only for GPU with FP16
    dynamic=False    # True if variable input size
)

print("✅ Exported to best.onnx")