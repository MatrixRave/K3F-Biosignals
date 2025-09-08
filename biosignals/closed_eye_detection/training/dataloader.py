
# Import the necessary module from the Python Imaging Library (PIL).
from PIL import ImageFile
# Define the list of file names
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image as PIL_Image

from glob import glob



def convert_image_to_greyscale(sample):
    image = PIL_Image.open(sample["image"]).convert("L")  # "L" = Greyscale
    return {"image": image}


def load_image_filepaths_as_dataframe(dataset_dir):
    print('Loading image files...')
    image_files = glob(os.path.join(dataset_dir, '*.jpg'))
    # use https://huggingface.co/docs/datasets/image_load for reference
    image_dict = {}
    file_names = []
    labels = []

    all_images = glob(os.path.join(dataset_dir, '*/*.png'))
    for fp in tqdm(all_images):
        label = os.path.basename(fp).replace('.png', '').split('_')[-1]  # open / closed
        labels.append(label)  # Add the label to the list
        file_names.append(fp)  # Add the file path to the list
        print(fp, label)

    print('Loaded imagefiles', len(file_names), 'images', len(labels), 'labels')

    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    print(df.shape)
    print(df.head())
    return df
