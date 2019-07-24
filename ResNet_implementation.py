import pandas as pd
import tensorflow as tf
import os
import logging 
import ResNet
import sklearn.model_selection as sk
from keras.utils import to_categorical, plot_model
import numpy as np



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
    all_images_list = pd.DataFrame(os.listdir(images_path), columns=['file'])
    # remove file extensions to get image id
    for index, row in all_images_list.iterrows():
        row['file'] = row['file'].replace('.jpg', '')
    # all_images_id = [id.replace('.jpg', '') for id in all_images_list['file']]

    images_to_load = all_images_list[all_images_list.isin(images_to_load_csv['id'])]

    for index, row in all_images_list.iterrows():
        row['file'] = row['file']+'.jpg'

    # reattach file extension to load in images
    # all_images_id_extension = [id.append('.jpg') for id in images_to_load]

    #images = [tf.read_file(images_path/file) for file in all_images_id_extension]
    images = pd.DataFrame()   
    for index, row in all_images_list.iterrows():
        images.add(tf.read_file('{}/{}'.format(images_path, row['file'])))
    return images


def copy_chosen_images(images_csv_path, images_path):
    images_to_load_csv = pd.read_csv(images_csv_path)

    # create list of all files
    all_images_list = os.listdir(images_path)
    logger.debug('getting list of images')
    # remove file extensions to get image id
    all_images_id = [id.replace('.jpg', '') for id in all_images_list]
    logger.debug('removing .jpg to get image id')
    logger.debug('creating list of selected images')
    images_to_load = [all_images_id.isin(images_to_load_csv['id'])]
   
    logger.debug('reconstructing file extension')
    # reattach file extension to load in images
    all_images_id_extension = [id.append('.jpg') for id in images_to_load]

    # move files listed in all_images_id_extension to new folder
    logger.debug('checking if directory exists')
    if not os.path.exists('reduced_dataset'):
        os.system('mkdir reduced_dataset')

    for image in all_images_id_extension:
        os.system('cp {}/{} reduced_dataset/{}'.format(images_folder, image, image))
        logger.debug('moving {}'.format(image))


def get_classes(data_csv):
    # Get all landmark ids that have at least 20 examples
    example_counts = data_csv.groupby('landmark_id').count()
    # separate the landmark id from the rest of the data
    example_indexes = example_counts.index.values
    return example_indexes

logger = create_logger('move_selected_images.log')
reduced_csv = "20_examples.csv"
data_csv = pd.read_csv(reduced_csv)
images_folder = "train"
# copy_chosen_images(reduced_csv, images_folder)
# labels = get_classes(data_csv)
# labels = labels.reshape((1, -1))
images = load_images(reduced_csv, images_folder)
images = images.values.reshape((1, -1))
# images = images.reshape((1,-1))
labels = data_csv.landmark_id
labels = labels.values.reshape((1, -1))

num_classes = 52584  # remove hardcoded number, use unique values in reduced csv
stages = (3, 4, 6)
filters = (64, 128, 256, 512)

# Set parameters for learning
batch_size = 128
# test_size = 256
img_size = 224

# Placeholder variable for input images
# Shape is [None, img_size, img_size, 1]
# None specifies that the tensor can hold an arbitrary number of images
X = tf.placeholder("float", [None, img_size, img_size, 1])

# Placeholder Y for the labels associated correctly with input images in placeholder X
# Shape is [None, num_classes]
# None specifies that the tensor can hold an arbitrary number of labels, each of length num_classes
Y = tf.placeholder("float", [None, num_classes])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# Labels need to be coded
# labels_ = np.zeros((images.shape[0], labels))
# labels_[np.arange(images.shape[0]), labels] = 1

trainX, trainY, testX, testY = sk.train_test_split(images, labels, test_size=0.25, random_state=42)



# split data into train and test set
# Measure performance using cross entropy. Alwyas positive and equal to 0 if predicted == output.
# Want to minimise the cross-entropy by changing layer variables
# Cross-entropy function calculates softmax internally so use output of model(...) directly
# py_x = ResNet.ResNet.build(img_size, img_size, 3, num_classes, stages, filters)
model = ResNet.ResNet()
model.build(img_size, img_size, 3, num_classes, stages, filters)

model.summary()
plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is a good metric for classification tasks
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the network
model.fit(trainX, trainY, epochs=20, batch_size=batch_size)
# validate the model on test dataset to determine generalization
loss, acc = model.evaluate(testX, testY, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))






