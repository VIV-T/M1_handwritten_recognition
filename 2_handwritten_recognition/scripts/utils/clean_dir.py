
from pathlib import Path
import glob
import os

# Use relative path from scripts folder to data folder
script_dir = Path(__file__).parent
data_folder = script_dir.parent / 'data'
raw_data_folder = script_dir.parent / 'data' / 'raw'
characters_folder = data_folder / 'characters'

def clean_dir(dir : str):
    characters_images = glob.glob(f'{dir}/*')

    for image in characters_images :
        os.remove(image)

    print(f"Files from {dir} removed succefully")
