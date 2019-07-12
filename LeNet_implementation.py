import tensorflow as tf
import numpy as np
import os
# import pathlib
import csv
import pandas as pd

#  resized_image_dir = pathlib.Path("E:\\Documents\\CompSci\\project\\trainset\\resized")
train_csv = "train.csv"
n_examples_file = "20_examples.csv"
# batch_csv_file = "trainset\\resized\\trial_batch_ids.csv"

# Get the number of images in the data set
def get_number_images(data_dir):
    return len(os.listdir(data_dir))


# Load images into dataframe
def load_images_to_list(data_dir):
    all_images = []
    for image in data_dir.iterdir():
        all_images.append(image)
    return all_images


def create_resized_batch_csv(csv_file, batch_dir, batch_csv_file):
    # Loads in the csv containing the image id and the landmark id contained within that image
    image_landmark_ids = pd.read_csv(csv_file)

    # Want to get all ids that correspond to an image in our small trial set
    present_images = pd.DataFrame()

    # build list of resized image ids and then we will add these to a new array
    image_ids = []
    for file in os.listdir(batch_dir):
        # image id is the file name with file extension removed
        temp_name = file.replace(".jpg", "")
        image_ids.append(temp_name)

    # grab the corresponding lines and add them to our new data frame
    present_images = present_images.append(image_landmark_ids.loc[image_landmark_ids['id'].isin(image_ids)])
    present_images.to_csv(batch_csv_file, index=None)


# create_resized_batch_csv(csv_file, resized_image_dir, batch_csv_file)

# batch_landmarks_ids = pd.read_csv(batch_csv_file)
# batch_landmarks_ids.groupby('landmark_id').count()


def generate_csv_n_examples(min_number_examples, n_examples_csv, source_csv):
    df_min_20 = pd.DataFrame()

    images_csv = pd.read_csv(source_csv)

    # Get all landmark ids that have at least 20 examples
    example_counts = images_csv.groupby('landmark_id').count()
    example_counts = example_counts[example_counts.id >= min_number_examples]
    # separate the landmark id from the rest of the data
    example_counts = example_counts.index.values

    #  Get landmark id for current row, attempt to add to id to new dataframe (id, count),
    #  if already exists incrememnt count
    for index, row in images_csv.iterrows():
        if row['landmark_id'] in example_counts:
            df_min_20 = df_min_20.append(row)

    outfile = open(n_examples_csv, 'wb')	
    df_min_20.to_csv(outfile, index=None)
    outfile.close()    


generate_csv_n_examples(20, n_examples_file, train_csv)











