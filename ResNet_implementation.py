import pandas as pd
import tensorflow as tf
import pathlib
import os


# load in images
def load_images(images_csv_path, images_path):
    images_to_load_csv = pd.read_csv(images_csv_path)

    # create list of all files
    all_images_list = os.listdir(images_path)
    # remove file extensions to get image id
    all_images_id = [id.replace('.jpg', '') for id in all_images_list]

    images_to_load = [all_images_id.isin(images_to_load_csv['id'])]

    # reattach file extension to load in images
    all_images_id_extension = [id.append('.jpg') for id in all_images_id]

    images = [tf.read_file(images_path/file) for file in all_images_id_extension]
    return images


reduced_csv = "20_examples.csv"
images_folder = "train"
load_images(reduced_csv, images_folder)







