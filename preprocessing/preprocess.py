import cv2
import numpy as np

def load_image(image_path):
    image=cv2.imread(image_path) # type: ignore
    if image is None:
        raise ValueError(f"Image not found at given path")
    return image
def preprocess_image(image):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray, (5,5), 0)
    return blur
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    # Apply filter2D to sharpen the image using the kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def detect_edges(image):
    # Use Canny edge detection on the grayscale image
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges
