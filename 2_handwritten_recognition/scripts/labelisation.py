import glob
from pathlib import Path
import string
import pandas as pd


script_dir = Path(__file__).parent
data_folder = script_dir.parent / 'data'
characters_folder = data_folder / 'characters'

def labelize_data() :
    dict_labelized = {"filepath":[], "labels":[]}

    for letter in string.ascii_lowercase:
        
        images = glob.glob(f'{characters_folder}/{letter}/*')

        for img in images :
            dict_labelized['filepath'].append(img)
            dict_labelized['labels'].append(letter)

    df_labelized = pd.DataFrame(dict_labelized)
    df_labelized.to_csv(f"{data_folder}/labels.csv", encoding="utf-8", index=False)
    print("CSV created")