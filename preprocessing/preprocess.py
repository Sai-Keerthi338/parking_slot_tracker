import cv2
import numpy as np
import os

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def find_parking_slots(edge_image, original_image, min_area=800, aspect_ratio_range=(1.5, 4.5)):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output_image = original_image.copy()
    detected_slots = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Check if contour is rectangular
                x, y, w, h = cv2.boundingRect(contour)
                if w < 14 or h < 14:
                    continue
                ratio = w / float(h)
                if aspect_ratio_range[0] <= ratio <= aspect_ratio_range[1] and area / (w * h) > 0.35:
                    detected_slots.append((x, y, w, h))
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print(f"Found {len(contours)} contours, {len(detected_slots)} valid slots")


    return output_image, detected_slots


def classify_occupancy(image, slot_rects, threshold=110):
    output = image.copy()
    status_list = []
    occupied = 0

    for idx, (x, y, w, h) in enumerate(slot_rects, 1):
        roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
        white_ratio = cv2.countNonZero(binary) / (w * h)

        if white_ratio< 0.38:
            status = "empty"
            color = (0, 0, 255)
            occupied += 1
        else:
            status = "occupied"
            color = (0, 255, 0)

        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output, f"Slot {idx}: {status}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        status_list.append((idx, status))

    return output, status_list, occupied