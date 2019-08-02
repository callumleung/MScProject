import pandas as pd
import tensorflow as tf
import os
import logging 
import ResNet
import sklearn.model_selection as sk
import pickle
from keras.preprocessing import image

# import matplotlib.pyplot as plt
# from keras.utils import to_categorical
# from keras.utils import plot_model
import numpy as np

# cite https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/ 2/08/19
# label binarizer object containing class labels
# mode is either train or eval, in eval no augmentation is applied
# aug: if supplied will be applied before yeilding images and labels
def csv_image_generator(input_path, images_folder, batch_size, label_binarizer, mode="train", aug=None):
    # Open csv
    f = open(input_path, "r")

    # loop infinitely
    while True:
        # initialise images and labels
        images = []
        labels = []
        # loop while images is less than the batch size
        while len(images) < batch_size:
            # attempt to read next line of csv
            line = f.readline()

            # check if end of file has been reached
            if line == "":
                # return to the head of the file so that the stream does no end causing an error in the fit_batch method
                f.seek(0)
                line = f.readline()

                # if evaluating break from loop so as to not continuously fill batch from samples at head of file
                if mode == "eval":
                    break

            # extract label (landmark id) and load the image
            # line format in our csv is "file id, url, landmark id"
            line = line.strip().split(",")
            # label = landmark_id, the 3rd entry of the line
            label = line[2]
            # append .jpg to the image id to load the image
            image_uri = str(line[0]) + ".jpg"
            image_path = "{}/{}".format(images_folder, image_uri)
            img = image.load_img(image_path)
            # convert to a usable array
            img = image.img_to_array(img)

            # add to current working batches lists
            images.append(img)
            labels.append(label)

            # One hot encode labels
            labels = label_binarizer.transform(np.array(labels))

            # deal with data augmentation
            if aug is not None:
                (images, labels) = next(aug.flow(np.array(images), labels, batch_size=batch_size))

            # finally yield images to calling function
            yield(np.array(images), labels)




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
    images_list = []
    columns = ['id']
    for files in os.listdir(images_path):
        images_list.append(str(files))
            
    images_list = pd.DataFrame(images_list, columns=columns)
    print("printing images_list first time\n")
    print(images_list.head())
    # all_images_list = pd.DataFrame(os.listdir(images_path), columns=['file'])
    # remove file extensions to get image id
    for index, row in images_list.iterrows():
        replace_id = str(row['id']).replace(".jpg", "")
        images_list.set_value(index, 'id', replace_id)
    # all_images_id = [id.replace('.jpg', '') for id in all_images_list['file']]
    # grab ids that are in the selected batch
    print("printing images_list with .jpg replaced\n")
    print(images_list.head())

    images_to_load = images_list[images_list.id.isin(images_to_load_csv.id)]
    # images_to_load = images_list[images_list.isin(images_to_load_csv['ids'])]

    # reattach file ending to use to load image
    for index, row in images_to_load.iterrows():
        file_address = str(row['id']) + ".jpg"
        images_to_load.set_value(index, 'id', file_address)
    # reattach file extension to load in images
    # all_images_id_extension = [id.append('.jpg') for id in images_to_load]

    # TODO: load images in batches rather than all at once
    # images = [tf.read_file(images_path/file) for file in all_images_id_extension]
    images = []
    print("prtingin images_to_load")
    print(images_to_load.head())
    # I think this needs to be changed to iterows
    for index, row in images_to_load.iterrows():
        if row['id'] != 'id':
            path = '{}/{}'.format(images_path, row['id'])
            # print(path)
            temp_img = image.load_img(path)
            temp_img_array = image.img_to_array(temp_img)
            images.append(temp_img_array)
            temp_img.close()

    print("loaded Images shape:")
    print(len(images))

    return_df = pd.DataFrame(images, columns=['images'])
    print("printing return shape")
    print(return_df.shape)
    return return_df


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
print("about to load images")
images = load_images(reduced_csv, images_folder)
print("printing images shape")
print(images.shape)
# images = images.values.reshape((1, -1))

# convert images into useable form
# for imag in images:
#     imag = images.img_to_array(imag)

#
# images = images.reshape((1,-1))
labels = data_csv.landmark_id
# labels = labels.values.reshape((1, -1))
print("printing labels shape")
print(labels.shape)

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

# trainX, trainY, testX, testY = sk.train_test_split(images, labels, test_size=0.25, random_state=42)



# split data into train and test set
# Measure performance using cross entropy. Alwyas positive and equal to 0 if predicted == output.
# Want to minimise the cross-entropy by changing layer variables
# Cross-entropy function calculates softmax internally so use output of model(...) directly
# py_x = ResNet.ResNet.build(img_size, img_size, 3, num_classes, stages, filters)
model = ResNet.ResNet()
model = model.build(img_size, img_size, 3, num_classes, stages, filters)

# model.summary()
# plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

# use of adam optimizer
# accuracy is a good metric for classification tasks
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the network
# TODO: replace with fit_generator
# performing data argumentation by training image generator
dataAugmentaion = image.ImageDataGenerator(rotation_range=30, zoom_range=0.20,
                                           fill_mode="nearest", shear_range=0.20, horizontal_flip=True,
                                           width_shift_range=0.1, height_shift_range=0.1)
model_history = model.fit_generator
#model_history = model.fit(images, labels, epochs=20, batch_size=batch_size)

# Plot training & validation accuracy values
# from https://keras.io/visualization/
# plt.plot(model_history.history['acc'])
# plt.plot(model_history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig("ResNet_train_history.png")

with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(model_history.history, file_pi)
# validate the model on test dataset to determine generalization
# TODO: change data and labels to test set
loss, acc = model.evaluate(images, labels, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))






