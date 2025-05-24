from preprocessing.preprocess import load_image, preprocess_image, sharpen_image, detect_edges
import matplotlib.pyplot as plt
import cv2

def main():
    image_path="data/parking_zone.jpg"
    image=load_image(image_path)
    preprocessed=preprocess_image(image)

    sharpened=sharpen_image(preprocessed) 
    edges=detect_edges(preprocessed)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

if __name__=="__main__":
    main()
    