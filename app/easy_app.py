import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🅿️ Parking")

f = st.file_uploader("Image", type=["jpg", "png"])
if f:
    r = YOLO("best.pt")(Image.open(f))[0]
    
    # Толщина рамки, цвета, шрифт
    img = r.plot(
        line_width=2,           # толщина рамки (по умолч. ~3)
        font_size=0.5,          # размер шрифта метки
        labels=True,            # показывать метки
        conf=True,              # показывать confidence
        boxes=True,             # показывать рамки
    )
    st.image(img)