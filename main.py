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



if __name__=="__main__":
    main()
    
