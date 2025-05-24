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
def find_parking_slots(edge_image, original_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    slot_count = 0
    output_image = original_image.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Skip small noise shapes
            x, y, w, h = cv2.boundingRect(contour)
            if 1.5 < w/h < 5:  # Likely a rectangle slot
                slot_count += 1
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output_image, f"Slot {slot_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return output_image
