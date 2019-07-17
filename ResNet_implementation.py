import pandas as pd
import tensorflow as tf
import pathlib
import os

def create_logger(filename,
                  logger_name='logger',
                  file_fmt='%(asctime)s %(levelname)-8s: %(message)s',
                  console_fmt='%(asctime)s | %(message)s',
                  file_level=logging.DEBUG,
                  console_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_fmt = logging.Formatter(file_fmt)
    log_file = logging.FileHandler(filename)
    log_file.setLevel(file_level)
    log_file.setFormatter(file_fmt)
    logger.addHandler(log_file)

    console_fmt = logging.Formatter(console_fmt)
    log_console = logging.StreamHandler()
    log_console.setLevel(logging.DEBUG)
    log_console.setFormatter(console_fmt)
    logger.addHandler(log_console)

    return logger

# load in images
def load_images(images_csv_path, images_path):
    images_to_load_csv = pd.read_csv(images_csv_path)

    # create list of all files
    all_images_list = os.listdir(images_path)
    # remove file extensions to get image id
    all_images_id = [id.replace('.jpg', '') for id in all_images_list]

    images_to_load = [all_images_id.isin(images_to_load_csv['id'])]

    # reattach file extension to load in images
    all_images_id_extension = [id.append('.jpg') for id in images_to_load]

    images = [tf.read_file(images_path/file) for file in all_images_id_extension]
    return images


def copy_chosen_images(images_csv_path, images_path):
    images_to_load_csv = pd.read_csv(images_csv_path)

    # create list of all files
    all_images_list = os.listdir(images_path)
    # remove file extensions to get image id
    all_images_id = [id.replace('.jpg', '') for id in all_images_list]

    images_to_load = [all_images_id.isin(images_to_load_csv['id'])]

    # reattach file extension to load in images
    all_images_id_extension = [id.append('.jpg') for id in images_to_load]

    # move files listed in all_images_id_extension to new folder
    if not os.path.exists('reduced_dataset'):
        os.system('mkdir reduced_dataset')

    for image in all_images_id_extension:
        os.system('cp {}/{} reduced_dataset/{}'.format(images_folder, image, image))
        logger.debug('moving {}'.format(image))


logger = create_logger('move_selected_images.log')
reduced_csv = "20_examples.csv"
images_folder = "train"
copy_chosen_images(reduced_csv, images_folder)



