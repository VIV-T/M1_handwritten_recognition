import cv2
import numpy as np
from pathlib import Path
import glob
import os
import string 

from utils.clean_dir import clean_dir
from utils.extract_contour import extract_contours

# Use relative path from scripts folder to data folder
script_dir = Path(__file__).parent
data_folder = script_dir.parent / 'data'
raw_data_folder = script_dir.parent / 'data' / 'raw'
characters_folder = data_folder / 'characters'



def build_db():
    for letter in string.ascii_lowercase:
        clean_dir(dir=f'{characters_folder}/{letter}')
    print("Database cleaned succefully !")

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

        ## Exctract the characters from the raw image
        gray, contours = extract_contours(raw_file=raw_file)

        # Loop through the contours and save each character as a separate image
        for i, contour in enumerate(contours):
            # Get bounding box for each contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Filter out small contours that may be noise
            if THRESHOLD < w and THRESHOLD < h :  # Ajust the threshold as needed
                # Extract the character from the grayscale image
                char_img = gray[y:y+h, x:x+w]

                # Save the character image to the corresponding folder
                cv2.imwrite(f'{letter_folder}/char_{i}.png', char_img)

        print(f"Nombre de caractères isolés pour la lettre {letter} : {len([c for c in contours if THRESHOLD < cv2.boundingRect(c)[2] and THRESHOLD < cv2.boundingRect(c)[3]])},    THRESHOLD : {THRESHOLD}")

        print("\n")
        print("Add Manual deletion of wrong extracted characters.") # like partial characters extracted.
        print("\n")