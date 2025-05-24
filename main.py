from preprocessing.preprocess import load_image, preprocess_image
import matplotlib.pyplot as plt
import cv2

def main():
    image_path="data/parking_zone.jpg"
    image=load_image(image_path)
    preprocessed=preprocess_image(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image (Grayscale + Blur)")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
    