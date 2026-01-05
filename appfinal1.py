import streamlit as st
import cv2
import numpy as np

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Virtual Paint (Cloud Preview)", layout="centered")

st.title("Virtual Paint 🎨")

st.markdown("""
This is the **cloud preview** of my Virtual Paint project.

- It involves real-time webcam to demonstrate color based object detection & masking
- **Cloud version** demonstrates **color detection pipeline** on a captured image.
""")
#TRY WITH A YELLOW OBJECT!

# -------------------- CLOUD DEMO --------------------
st.subheader("Color Detection Demo (Cloud-Safe)")

img = st.camera_input("Capture a frame")

if img is not None:
    # Convert Streamlit image to OpenCV format (FIXED)
    file_bytes = np.asarray(bytearray(img.getvalue()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Mirror image for natural view
    frame = cv2.flip(frame, 1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --------- COLOR RANGE (YELLOW OBJECT EXAMPLE) ---------
    lower = np.array([20, 100, 100])
    upper = np.array([35, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding box around largest detected object
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    st.image(frame, channels="BGR", caption="Detected Color Object")
    st.image(mask, caption="HSV Mask")
