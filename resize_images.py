# credit https://www.kaggle.com/sermakarevich/download-resize-clean-12-hours-44gb
import logging
import math
import os
import subprocess
import shutil
from multiprocessing import Pool

from PIL import Image

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

def resize_folder_images(src_dir, dst_dir, size=224):
    if not os.path.isdir(dst_dir):
        logger.info("destination directory does not exist, creating destination directory.")
        os.makedirs(dst_dir)

    image_filenames = os.listdir(src_dir)
    count = 0
    for filename in image_filenames:
        dst_filepath = os.path.join(dst_dir, filename)
        src_filepath = os.path.join(src_dir, filename)
        if os.path.isfile(src_filepath):
            new_img = read_and_resize_image(src_filepath, size)
            if new_img is not None:
                new_img = new_img.convert("RGB")
                new_img.save(dst_filepath)
                count += 1
    logger.debug(f'{src_dir} files resized: {count}')


def read_and_resize_image(filepath, size):
    img = read_image(filepath)
    if img:
        img = resize_image(img, size)
    return img


def resize_image(img, size):
    if type(size) == int:
        size = (size, size)
    if len(size) > 2:
        raise ValueError("Size needs to be specified as Width, Height")
    return resize_contain(img, size)


def read_image(filepath):
    try:
        img = Image.open(filepath)
        return img
    except (OSError, Exception) as e:
        logger.debug("Can't read file {}".format(filepath))
        return None


def resize_contain(image, size, resample=Image.LANCZOS, bg_color=(255, 255, 255, 0)):
    img_format = image.format
    img = image.copy()
    img.thumbnail((size[0], size[1]), resample)
    background = Image.new('RGBA', (size[0], size[1]), bg_color)
    img_position = (
        int(math.ceil((size[0] - img.size[0]) / 2)),
        int(math.ceil((size[1] - img.size[1]) / 2))
    )
    background.paste(img, img_position)
    background.format = img_format
    return background.convert('RGB')


def delete_files_from_directory(file_dir):
    files = os.listdir(file_dir)
    for f in files:
        file = file_dir + '\\' + f
        if os.path.isfile(file):
            print(file)
            os.remove(file)


logger = create_logger('download.log')

# logger.debug(f'Resizing images')

images_folder = "E:\\Documents\\CompSci\\project\\trainset"
delete_files_from_directory(images_folder)
# resized_folder = "E:\\Documents\\CompSci\\project\\trainset\\resized"
# resize_folder_images(
#     src_dir=images_folder,
#     dst_dir=resized_folder,
#     size=224
# )



