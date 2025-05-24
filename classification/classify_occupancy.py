import cv2
import numpy as np
def classify_occupancy(image, slots, threshold=100):
    #to classify as slots as occupied or empty
    results= []

    for slot in slots:
        x, y, w, h = slot
        roi = image[y:y+h, x:x+w]  # Crop region of interest
        mean_val = np.mean(roi)

        if mean_val < threshold:
            results.append((slot, "occupied"))
        else:
            results.append((slot, "empty"))

    return results