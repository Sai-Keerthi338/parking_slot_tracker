import os
import cv2

from preprocessing.preprocess import load_image, preprocess_image,sharpen_image,detect_edges, find_parking_slots, classify_occupancy
def main():
    image_path = "data/parking_zone.jpg"
    image = load_image(image_path)
    preprocessed = preprocess_image(image)
    sharpened = sharpen_image(preprocessed)
    edges = detect_edges(preprocessed)
      # Use sharpened edges

    slot_image, detected_slots = find_parking_slots(edges, image)
    final_image, status, occupied = classify_occupancy(image, detected_slots)

    total = len(detected_slots)
    available = total - occupied

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(status, columns=["Slot Number", "Status"])
    df.loc[len(df.index)] = ["Total Slots", total]
    df.loc[len(df.index)] = ["Occupied", occupied]
    df.loc[len(df.index)] = ["Available", available]
    df.to_csv("outputs/slot_status.csv", index=False)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/original.jpg", image)
    cv2.imwrite("outputs/preprocessed.jpg", preprocessed)
    cv2.imwrite("outputs/sharpened.jpg", sharpened)
    cv2.imwrite("outputs/edges.jpg", edges)
    cv2.imwrite("outputs/slots.jpg", slot_image)
    cv2.imwrite("outputs/slot_status.jpg", final_image)

    print("Done. Check outputs folder.")

if __name__=="__main__":
    main()
