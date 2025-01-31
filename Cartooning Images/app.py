import streamlit as st
import cv2
import numpy as np
from PIL import Image

class Cartoonizer:
    def render(self, img_rgb):
        # Convert image to 8-bit format if necessary
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)  # Normalize if needed
        
        # Ensure image has 3 channels (RGB)
        if len(img_rgb.shape) == 2:  # Grayscale image
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:  # RGBA image
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        numDownSamples = 2
        numBilateralFilters = 50

        img_color = img_rgb.copy()
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 75, 75)

        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_cartoon = cv2.bitwise_and(img_color, img_edge)

        return img_cartoon


def main():
    st.title("Cartoonizer App 🎨🖌️")
    st.write("Upload an image to apply the cartoon effect!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Cartoonize"):
            cartoonizer = Cartoonizer()
            cartoon_img = cartoonizer.render(img_array)

            st.image(cartoon_img, caption="Cartoonized Image", use_column_width=True)

if __name__ == "__main__":
    main()
