from preprocessing.preprocess import load_image, preprocess_image, sharpen_image, detect_edges, find_parking_slots
from classification.classify_occupancy import classify_occupancy

import matplotlib.pyplot as plt
import cv2
import os

def main():
    image_path="data/parking_zone.jpg"
    image=load_image(image_path)
    preprocessed=preprocess_image(image)

    sharpened=sharpen_image(preprocessed) 
    edges=detect_edges(preprocessed)

    slot_image = find_parking_slots(edges, image)

  
    os.makedirs("outputs", exist_ok=True)

    # Saving images to outputs folder
    cv2.imwrite("outputs/original.jpg", image)
    cv2.imwrite("outputs/preprocessed.jpg", preprocessed)
    cv2.imwrite("outputs/sharpened.jpg", sharpened)
    cv2.imwrite("outputs/edges.jpg", edges)
    cv2.imwrite("outputs/slots.jpg", slot_image)

    print("All intermediate images saved to outputs/")

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    slot_image_rgb = cv2.cvtColor(slot_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Preprocessed Image (Grayscale + Blur)")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Sharpened Image")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Canny edges")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Detected Parking Slots")
    plt.imshow(slot_image_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    """

if __name__=="__main__":
    main()
    