import cv2
import numpy as np

def extract_contours(raw_file):
    # read the img
    image = cv2.imread(raw_file)

    # Convert to gray level.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add a blur to reduce noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Appliquer un seuil binaire (inversé pour avoir le texte en blanc)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Nettoyer l'image avec des opérations morphologiques
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Détecter les contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, contours