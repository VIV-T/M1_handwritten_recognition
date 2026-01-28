# Script to identify the characters of a word or sentence.
# Imports
import pandas as pd
from pathlib import Path
import glob
import os
import cv2

from utils.clean_dir import clean_dir
from utils.extract_contour import extract_contours
from utils.extract_features import features_extraction

# Use relative path from scripts folder to data folder
script_dir = Path(__file__).parent
data_folder = script_dir.parent / 'data'
words_folder = data_folder / 'words'
extracted_characters_folder = words_folder / "characters"

THRESHOLD = 40



def extract_characters(filepath : str) : 
    ## Exctract the characters from the raw image
    gray, contours = extract_contours(raw_file=filepath)

    # Trier les contours par position horizontale (x)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    number_of_char = 0

    # Parcourir les contours et isoler les caractères
    for i, contour in enumerate(contours):
        # Obtenir le rectangle englobant du contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Filtrer les contours trop petits ou trop grands
        if THRESHOLD < w and THRESHOLD < h :  # Ajuste ces valeurs selon la taille de tes caractères
            # Extraire le caractère
            char_img = gray[y:y+h, x:x+w]

            # Sauvegarder le caractère en tant qu'image
            if i < 10 :
                cv2.imwrite(f'{extracted_characters_folder}/char_0{i}.png', char_img)
            else :
                cv2.imwrite(f'{extracted_characters_folder}/char_{i}.png', char_img)
            number_of_char += 1

    print(f"This word contains {number_of_char} characters.")
    print("Extraction finished !\n")


def create_df_to_label():
    characters = glob.glob(f'{extracted_characters_folder}/*')

    dict_char = {"filepath" :[]}

    for character in characters :
        filename = os.path.basename(character) 
        filepath = os.path.join(extracted_characters_folder, filename)
        dict_char['filepath'].append(filepath)

    df_char = pd.DataFrame(dict_char)

    print("Dataframe created !\n")

    return df_char


# Main function
def prepare_new_data(filename : str = "lancegoat.jpg", features_extraction_method : str|None = None, prepare_new_data : bool = False) :
    filepath = os.path.join(words_folder, filename)
    
    # clean the folder containing the characters => need to be empty
    clean_dir(extracted_characters_folder)

    # characters extraction
    extract_characters(filepath=filepath)

    # create the df with the filepath to then extract the features
    df_char = create_df_to_label()

    # features extraction
    X_new, y_new = features_extraction(df_labels=df_char, method=features_extraction_method, prepare_new_data=prepare_new_data)

    return X_new