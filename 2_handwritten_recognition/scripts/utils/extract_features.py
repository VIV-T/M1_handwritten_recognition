# Imports 
import numpy as np
import os
from pathlib import Path
import glob
import pandas as pd

from utils.features_extraction_methods.HOG import extract_hog_features
from utils.features_extraction_methods.HU import extract_hu_features
from utils.features_extraction_methods.GEOMETRIC import extract_geometric_features
from utils.features_extraction_methods.eighteen import extract_handcrafted_features



# --------------------------------------------------
# CSV processing
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images_dir")
CSV_PATH = os.path.join(IMAGES_DIR, "labels.csv")

# Use relative path from scripts folder to data folder
script_dir = Path(__file__).parent.parent
data_folder = script_dir.parent / 'data'
characters_folder = data_folder / 'characters'
features_folder = data_folder / 'features'


# Main function
def features_extraction(df_labels : pd.DataFrame, method : str|None = None, prepare_new_data : bool = False) :
    try :
        df_labels["labels"]
        complete = True     # the df contains the data and the labels 
    except :
        complete = False    # the df contains only the data without labels.

    print("The df is complete (data + labels) : " + str(complete))
    
    features = []
    labels = []

    #### Part I
    # method choice, based on the parameter of the function : HOG, handcrafted features, ...
    if True :
        pass


    #### Part II
    # features extraction
    for index, row in df_labels.iterrows():
        print(row["filepath"])
        match method :
            case "HOG" :
                feat = extract_hog_features(row["filepath"])  
            case "HU" :
                feat = extract_hu_features(row["filepath"])   
            case "GEOMETRIC" :
                feat = extract_geometric_features(row["filepath"])  
            case "eighteen" :
                feat = extract_handcrafted_features(row["filepath"])  

        
        if feat is not None:
            features.append(feat)
            if complete :
                labels.append(row["labels"])

    X = np.array(features)
    y = np.array(labels)
    
    print("Features extraction finished !\n")

    if not prepare_new_data :
        # Save features and labels
        method_folder = features_folder / method
        os.makedirs(method_folder, exist_ok=True)
        np.save(f"{method_folder}/features_{method}.npy", X)
        np.save(f"{method_folder}/labels_{method}.npy", y)
        return True
    else :    
        return X, y