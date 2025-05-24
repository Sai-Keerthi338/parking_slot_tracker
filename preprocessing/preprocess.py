import cv2
def load_image(image_path):
    image=cv2.imread(image_path) # type: ignore
    if image is None:
        raise ValueError(f"Image not found at given path")
    return image
def preprocess_image(image):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray, (5,5), 0)
    return blur
