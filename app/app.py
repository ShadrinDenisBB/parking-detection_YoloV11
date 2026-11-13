import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd

# ═══════════════════════════════════════════════════════
# MODEL PATH
# ═══════════════════════════════════════════════════════
MODEL_PATH = "best.pt"

# Colors for status display
COLOR_CERTAIN = "#22BB44"  # Green — high confidence
COLOR_UNCERTAIN = "#FFAA00"  # Orange — low confidence
COLOR_OCCUPIED = "#FF4444"  # Red
COLOR_EMPTY = "#44BBFF"  # Blue

# ═══════════════════════════════════════════════════════
# FIXED THRESHOLDS (Variant B)
# ═══════════════════════════════════════════════════════
# Certain: high confidence, reliable detection
# Uncertain: lower confidence, detected but flagged
OCCUPIED_CERTAIN = 0.50
OCCUPIED_UNCERTAIN = 0.30

EMPTY_CERTAIN = 0.35
EMPTY_UNCERTAIN = 0.20


# ═══════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)


# ═══════════════════════════════════════════════════════
# UI SETUP
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Parking Detection", layout="wide")

st.title("🚗 Parking Space Detection — YOLOv11")
st.markdown("Real-time detection of vacant and occupied parking spaces")

# Sidebar — info only, no sliders
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Model:** YOLOv11n  
**Resolution:** 1280×1280  
**Classes:** occupied, empty  
**Thresholds:** Fixed (Variant B)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Detection tiers:**
- 🟢 **Certain** — high confidence
- 🟡 **Uncertain** — lower confidence, flagged
""")

# Load model
try:
    model = load_model()
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "📤 Upload image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded is None:
    st.info("👆 Upload a file to start detection")
    st.stop()

# ═══════════════════════════════════════════════════════
# IMAGE PROCESSING
# ═══════════════════════════════════════════════════════
if uploaded.type.startswith("image"):
    image = Image.open(uploaded).convert('RGB')
    img_array = np.array(image)

    with st.spinner("🔍 Analyzing..."):
        results = model(img_array, conf=0.20, imgsz=1280, verbose=False)

    res = results[0]

    # Categorize detections by confidence tier
    occupied_certain = []
    occupied_uncertain = []
    empty_certain = []
    empty_uncertain = []

    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0:  # occupied
            if conf >= OCCUPIED_CERTAIN:
                occupied_certain.append({"conf": conf, "box": box})
            elif conf >= OCCUPIED_UNCERTAIN:
                occupied_uncertain.append({"conf": conf, "box": box})
        elif cls_id == 1:  # empty
            if conf >= EMPTY_CERTAIN:
                empty_certain.append({"conf": conf, "box": box})
            elif conf >= EMPTY_UNCERTAIN:
                empty_uncertain.append({"conf": conf, "box": box})

    n_occ_cert = len(occupied_certain)
    n_occ_uncert = len(occupied_uncertain)
    n_emp_cert = len(empty_certain)
    n_emp_uncert = len(empty_uncertain)

    n_occupied = n_occ_cert + n_occ_uncert
    n_empty = n_emp_cert + n_emp_uncert
    n_total = n_occupied + n_empty

    # Plot all detections (both certain and uncertain)
    all_detections = (occupied_certain + occupied_uncertain +
                      empty_certain + empty_uncertain)

    if all_detections:
        # Custom plot with color-coded confidence
        annotated = img_array.copy()
        for item in all_detections:
            box = item["box"]
            conf = item["conf"]
            cls_id = int(box.cls[0])

            # Determine color based on certainty
            if cls_id == 0:  # occupied
                if conf >= OCCUPIED_CERTAIN:
                    color = (0, 200, 0)  # Green (BGR)
                    label = f"occupied {conf:.2f}"
                else:
                    color = (0, 165, 255)  # Orange (BGR)
                    label = f"occupied? {conf:.2f}"
            else:  # empty
                if conf >= EMPTY_CERTAIN:
                    color = (255, 100, 0)  # Blue (BGR)
                    label = f"empty {conf:.2f}"
                else:
                    color = (0, 165, 255)  # Orange (BGR)
                    label = f"empty? {conf:.2f}"

            # Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        annotated = img_array.copy()

    # ═══════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ═══════════════════════════════════════════════════
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📷 Detection Result")
        st.image(annotated, use_container_width=True)

        # Legend
        legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
        legend_col1.markdown("🟢 occupied (certain)")
        legend_col2.markdown("🟡 occupied? (uncertain)")
        legend_col3.markdown("🔵 empty (certain)")
        legend_col4.markdown("🟡 empty? (uncertain)")

    with col2:
        st.subheader("📊 Statistics")

        # Occupancy metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("🚗 Occupied", n_occupied)
        c2.metric("🅿️ Empty", n_empty)
        c3.metric("📍 Total", n_total)

        if n_total > 0:
            occ_rate = n_occupied / n_total * 100
            st.progress(occ_rate / 100, text=f"Occupancy: {occ_rate:.1f}%")

        # Detailed breakdown
        st.markdown("---")
        st.subheader("🔍 Breakdown by Confidence")

        breakdown = pd.DataFrame({
            "Status": ["Occupied (certain)", "Occupied (uncertain)",
                       "Empty (certain)", "Empty (uncertain)"],
            "Count": [n_occ_cert, n_occ_uncert, n_emp_cert, n_emp_uncert],
            "Threshold": [f"≥{OCCUPIED_CERTAIN}", f"≥{OCCUPIED_UNCERTAIN}",
                          f"≥{EMPTY_CERTAIN}", f"≥{EMPTY_UNCERTAIN}"]
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        # Bar chart
        st.markdown("---")
        chart_data = pd.DataFrame({
            "Category": ["Occupied", "Empty"],
            "Count": [n_occupied, n_empty]
        })
        st.bar_chart(chart_data.set_index("Category"))

        # All detections table
        st.markdown("---")
        st.subheader("📋 All Detections")

        details = []
        for item in occupied_certain:
            details.append({"Class": "occupied", "Confidence": f"{item['conf']:.3f}",
                            "Status": "✅ certain"})
        for item in occupied_uncertain:
            details.append({"Class": "occupied?", "Confidence": f"{item['conf']:.3f}",
                            "Status": "⚠️ uncertain"})
        for item in empty_certain:
            details.append({"Class": "empty", "Confidence": f"{item['conf']:.3f}",
                            "Status": "✅ certain"})
        for item in empty_uncertain:
            details.append({"Class": "empty?", "Confidence": f"{item['conf']:.3f}",
                            "Status": "⚠️ uncertain"})

        if details:
            df = pd.DataFrame(details)
            # Sort by confidence descending
            df["Confidence_float"] = df["Confidence"].astype(float)
            df = df.sort_values("Confidence_float", ascending=False)
            df = df.drop("Confidence_float", axis=1)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No detections above threshold")

# ═══════════════════════════════════════════════════════
# VIDEO PROCESSING
# ═══════════════════════════════════════════════════════
elif uploaded.type.startswith("video"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ Failed to open video")
        os.unlink(tfile.name)
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.subheader("🎬 Video Processing")
    st.write(f"{total} frames, {fps:.1f} FPS")

    target_fps = st.slider("Target FPS", 1, int(fps), min(2, int(fps)))
    frame_skip = max(1, int(fps / target_fps))

    progress = st.progress(0)
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    all_occ = []
    all_emp = []
    frame_count = 0

    stop = st.button("⏹️ Stop")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, conf=0.20, imgsz=1280, verbose=False)
            res = results[0]

            # Count with same thresholds
            occ = sum(1 for b in res.boxes
                      if int(b.cls[0]) == 0 and float(b.conf[0]) >= OCCUPIED_UNCERTAIN)
            emp = sum(1 for b in res.boxes
                      if int(b.cls[0]) == 1 and float(b.conf[0]) >= EMPTY_UNCERTAIN)

            all_occ.append(occ)
            all_emp.append(emp)

            # Plot with same logic as images
            annotated = frame_rgb.copy()
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf >= OCCUPIED_UNCERTAIN:
                    color = (0, 200, 0) if conf >= OCCUPIED_CERTAIN else (0, 165, 255)
                    label = f"occupied {conf:.2f}" if conf >= OCCUPIED_CERTAIN else f"occupied? {conf:.2f}"
                elif cls_id == 1 and conf >= EMPTY_UNCERTAIN:
                    color = (255, 100, 0) if conf >= EMPTY_CERTAIN else (0, 165, 255)
                    label = f"empty {conf:.2f}" if conf >= EMPTY_CERTAIN else f"empty? {conf:.2f}"
                else:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame_placeholder.image(annotated, use_container_width=True)
            stats_placeholder.write(
                f"Frame {frame_count}: Occupied {occ}, Empty {emp}"
            )
            progress.progress(min(frame_count / total, 1.0))

        frame_count += 1

    cap.release()
    os.unlink(tfile.name)

    # Final stats
    st.markdown("---")
    st.subheader("📈 Video Summary")
    if all_occ:
        avg_occ = np.mean(all_occ)
        avg_emp = np.mean(all_emp)
        max_occ = max(all_occ)
        min_emp = min(all_emp)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Occupied", f"{avg_occ:.1f}")
        c2.metric("Avg Empty", f"{avg_emp:.1f}")
        c3.metric("Max Occupied", max_occ)
        c4.metric("Min Empty", min_emp)

        chart_df = pd.DataFrame({
            "Frame": range(len(all_occ)),
            "Occupied": all_occ,
            "Empty": all_emp
        })
        st.line_chart(chart_df.set_index("Frame"))

st.caption("Parking Space Detection | YOLOv11 | Computer Vision Project")