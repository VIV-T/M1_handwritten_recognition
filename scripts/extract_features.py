# Imports 
import numpy as np
import os
from pathlib import Path
import glob
import pandas as pd

from features_extraction_methods.HOG import extract_hog_features
from features_extraction_methods.HU import extract_handcrafted_features



# --------------------------------------------------
# CSV processing
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images_dir")
CSV_PATH = os.path.join(IMAGES_DIR, "labels.csv")

# Use relative path from scripts folder to data folder
script_dir = Path(__file__).parent
data_folder = script_dir.parent / 'data'
characters_folder = data_folder / 'characters'
features_folder = data_folder / 'features'


# Main function
def features_extraction(df_labels : pd.DataFrame, method : str|None = None) :
    features = []
    labels = []

    #### Part I
    # method choice, based on the parameter of the function : HOG, handcrafted features, ...
    if True :
        pass


    #### Part II
    # features extraction
    for index, row in df_labels.iterrows():
        feat = extract_hog_features(row["filepath"])   ### To modify.
        if feat is not None:
            features.append(feat)
            labels.append(row["labels"])


    X = np.array(features)
    y = np.array(labels)

    # print("X shape:", X.shape)  # (N_images, HOG_dim)
    # print("y shape:", y.shape)

    # np.save(f"{features_folder}/features_HOG.npy", X)
    # np.save(f"{features_folder}/labels_HOG.npy", y)

    print("Number of samples:", len(labels))

    return X, y