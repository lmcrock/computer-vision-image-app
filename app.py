import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(
    page_title="Computer Vision Image Lab",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Computer Vision Image Lab")
st.write(
    "Upload an image and experiment with image-processing techniques "
    "using OpenCV. Adjust the controls to see the results instantly."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    pil_image = Image.open(uploaded_file).convert("RGB")
    original = np.array(pil_image)

    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    st.sidebar.header("Image Controls")

    brightness = st.sidebar.slider(
        "Brightness",
        min_value=-100,
        max_value=100,
        value=0
    )

    contrast = st.sidebar.slider(
        "Contrast",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    processing_mode = st.sidebar.selectbox(
        "Processing Mode",
        [
            "Brightness & Contrast",
            "Grayscale",
            "Global Threshold",
            "Adaptive Threshold",
            "Histogram Equalization",
            "HSV Color Segmentation"
        ]
    )

    # Brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(
        img,
        alpha=contrast,
        beta=brightness
    )

    processed = adjusted

    # Processing modes
    if processing_mode == "Grayscale":
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        processed = gray

    elif processing_mode == "Global Threshold":
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        threshold_value = st.sidebar.slider(
            "Threshold Value",
            min_value=0,
            max_value=255,
            value=120
        )

        _, processed = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

    elif processing_mode == "Adaptive Threshold":
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    elif processing_mode == "Histogram Equalization":
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

        # Equalize brightness only so colors are preserved
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        processed = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

    elif processing_mode == "HSV Color Segmentation":
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

        hue_min = st.sidebar.slider(
            "Minimum Hue",
            min_value=0,
            max_value=179,
            value=35
        )

        hue_max = st.sidebar.slider(
            "Maximum Hue",
            min_value=0,
            max_value=179,
            value=80
        )

        saturation_min = st.sidebar.slider(
            "Minimum Saturation",
            min_value=0,
            max_value=255,
            value=50
        )

        value_min = st.sidebar.slider(
            "Minimum Value",
            min_value=0,
            max_value=255,
            value=50
        )

        lower = np.array(
            [hue_min, saturation_min, value_min],
            dtype=np.uint8
        )

        upper = np.array(
            [hue_max, 255, 255],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower, upper)

        processed = cv2.bitwise_and(
            adjusted,
            adjusted,
            mask=mask
        )

    # Display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)

    with col2:
        st.subheader("Processed Image")

        if len(processed.shape) == 2:
            st.image(
                processed,
                clamp=True,
                use_container_width=True
            )
        else:
            processed_rgb = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2RGB
            )
            st.image(
                processed_rgb,
                use_container_width=True
            )

    st.divider()

    st.subheader("Computer Vision Concepts Used")

    st.write(
        """
        This application demonstrates several computer vision concepts:

        - **Brightness and contrast adjustment** changes pixel intensity values.
        - **Grayscale conversion** reduces a color image to one intensity channel.
        - **Global thresholding** separates pixels using a selected threshold value.
        - **Adaptive thresholding** calculates thresholds based on local image regions.
        - **Histogram equalization** improves brightness and contrast while preserving color by equalizing the HSV value channel.
        - **HSV color segmentation** creates a mask from hue, saturation, and value ranges and uses it to isolate selected colors.
        """
    )

else:
    st.info("👆 Upload a JPG or PNG image to begin.")
