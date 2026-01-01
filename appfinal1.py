import streamlit as st
import cv2
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Virtual Paint",
    layout="centered"
)

st.title("Virtual Paint 🎨")
st.caption("Draw in the air using color-based object tracking (OpenCV)")

# --------------------------------------------------
# Function: Get HSV limits for a BGR color
# --------------------------------------------------
def get_limits(color):
    """
    Converts a BGR color to HSV and returns
    lower and upper HSV bounds.
    """
    c = np.uint8([[color]])
    hsv_color = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    h = int(hsv_color[0][0][0])

    lower = np.array([max(h - 10, 0), 100, 100], dtype=np.uint8)
    upper = np.array([min(h + 10, 179), 255, 255], dtype=np.uint8)

    return lower, upper


# --------------------------------------------------
# Controls
# --------------------------------------------------
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

# Drawing color (BGR)
draw_color = [0, 255, 255]  # Yellow object
paint_color = (0, 0, 255)   # Red paint on canvas

# --------------------------------------------------
# Initialize session state (important!)
# --------------------------------------------------
if "cap" not in st.session_state:
    st.session_state.cap = None

if "canvas" not in st.session_state:
    st.session_state.canvas = None


# --------------------------------------------------
# Main logic (NO infinite loop)
# --------------------------------------------------
if run:

    # Start webcam only once
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    cap = st.session_state.cap
    ret, frame = cap.read()

    if not ret:
        st.warning("Unable to access webcam")
        st.stop()

    # Flip for natural movement
    frame = cv2.flip(frame, 1)

    # Create canvas once
    if st.session_state.canvas is None:
        st.session_state.canvas = np.zeros_like(frame)

    canvas = st.session_state.canvas

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask based on color
    lower, upper = get_limits(draw_color)
    mask = cv2.inRange(hsv, lower, upper)

    # Reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2

            # Draw on canvas
            cv2.circle(canvas, (cx, cy), 5, paint_color, -1)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # Merge canvas with frame
    output = cv2.add(frame, canvas)

    # Display
    FRAME_WINDOW.image(output, channels="BGR")

else:
    # Cleanup when checkbox is turned off
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    st.session_state.canvas = None
    # Show blank image when not running