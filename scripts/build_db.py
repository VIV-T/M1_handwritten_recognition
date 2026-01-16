import cv2
import numpy as np
from pathlib import Path
import glob
import os

from clean_db import clean_db

# Use relative path from scripts folder to data folder
script_dir = Path(__file__).parent
data_folder = script_dir.parent / 'data'
raw_data_folder = script_dir.parent / 'data' / 'raw'
characters_folder = data_folder / 'characters'



def build_db():
    clean_db()
    raw_files = glob.glob(f'{raw_data_folder}/*')

    for raw_file in raw_files :
        # Letter extraction and path creation
        filename = os.path.basename(raw_file) 
        letter = os.path.splitext(filename)[0]
        letter_folder = characters_folder / letter

        if letter == "k" or letter == "y" :
            THRESHOLD = 50
        elif letter == "m" or letter == "n" :
            THRESHOLD = 25
        else : 
            THRESHOLD = 40

        # exctract the characters from the raw image
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

        # Parcourir les contours et isoler les caractères
        for i, contour in enumerate(contours):
            # Obtenir le rectangle englobant du contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Filtrer les contours trop petits ou trop grands
            if THRESHOLD < w and THRESHOLD < h :  # Ajuste ces valeurs selon la taille de tes caractères
                # Extraire le caractère
                char_img = gray[y:y+h, x:x+w]

                # Sauvegarder le caractère en tant qu'image
                cv2.imwrite(f'{letter_folder}/char_{i}.png', char_img)

        print(f"Nombre de caractères isolés pour la lettre {letter} : {len([c for c in contours if THRESHOLD < cv2.boundingRect(c)[2] and THRESHOLD < cv2.boundingRect(c)[3]])},    THRESHOLD : {THRESHOLD}")

        print("\n")
        print("Add Manual deletion of wrong extracted characters.") # like partial characters extracted.
        print("\n")