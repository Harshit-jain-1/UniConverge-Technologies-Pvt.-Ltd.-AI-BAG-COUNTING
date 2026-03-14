
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

st.set_page_config(page_title="Warehouse Management System", layout="wide")

model = YOLO("yolov8n.pt")

st.markdown(
"""
<h1 style='text-align:center;color:yellow;'>Warehouse Management System</h1>
<h3 style='text-align:center;color:orange;'>Bag Counting Management System</h3>
""",
unsafe_allow_html=True
)

bag_in = {"gate1":0,"gate2":0,"gate3":0}
bag_out = {"gate1":0,"gate2":0,"gate3":0}


def process_video(video_file, frame_holder, in_metric, out_metric, gate):

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    previous_positions = {}
    counted_ids = set()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(640,480))

        height,width,_ = frame.shape
        red_line = width // 2

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box,track_id in zip(boxes,ids):

                x1,y1,x2,y2 = box
                center_x = int((x1+x2)/2)

                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)

                if track_id not in previous_positions:
                    previous_positions[track_id] = center_x

                else:

                    prev_x = previous_positions[track_id]

                    if prev_x > red_line and center_x <= red_line and track_id not in counted_ids:
                        bag_in[gate] += 1
                        counted_ids.add(track_id)

                    elif prev_x < red_line and center_x >= red_line and track_id not in counted_ids:
                        bag_out[gate] += 1
                        counted_ids.add(track_id)

                    previous_positions[track_id] = center_x

        cv2.line(frame,(red_line,0),(red_line,height),(0,0,255),3)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_holder.image(frame, channels="RGB", use_container_width=True)

        in_metric.metric("Loading Bags", bag_in[gate])
        out_metric.metric("Unloading Bags", bag_out[gate])

        time.sleep(0.03)

    cap.release()


# Dashboard Layout
gate1,gate2,gate3 = st.columns(3)

with gate1:
    st.subheader("Gate Number : 01")
    video1 = st.file_uploader("Upload Gate1 Video", type=["mp4","avi","mov"])
    frame_holder1 = st.empty()
    in_metric1 = st.empty()
    out_metric1 = st.empty()

with gate2:
    st.subheader("Gate Number : 02")
    video2 = st.file_uploader("Upload Gate2 Video", type=["mp4","avi","mov"])
    frame_holder2 = st.empty()
    in_metric2 = st.empty()
    out_metric2 = st.empty()

with gate3:
    st.subheader("Gate Number : 03")
    video3 = st.file_uploader("Upload Gate3 Video", type=["mp4","avi","mov"])
    frame_holder3 = st.empty()
    in_metric3 = st.empty()
    out_metric3 = st.empty()


# Process videos after upload
if video1:
    process_video(video1, frame_holder1, in_metric1, out_metric1, "gate1")

if video2:
    process_video(video2, frame_holder2, in_metric2, out_metric2, "gate2")

if video3:
    process_video(video3, frame_holder3, in_metric3, out_metric3, "gate3")


st.markdown("---")

st.markdown(
"<h2 style='text-align:center;color:cyan'>IOT Parameters Monitoring</h2>",
unsafe_allow_html=True
)

iot1,iot2 = st.columns(2)

with iot1:
    st.metric("Temperature","28 C")
    st.metric("Humidity","62 %")
    st.metric("Phosphine Gas Level","Safe")

with iot2:
    st.metric("Smoke & Fire Status","Normal")
    st.metric("Gate Status","Closed")
