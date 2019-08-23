import pandas as pd

import os
import logging 
import ResNet

from keras.preprocessing import image
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer

import numpy as np


# based on https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/ 2/08/19
# with alterations to load from images rather than csv files
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
            if line[0] != 'id':
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


# load in images
# now deprecated as there are far too many images to load into memory at once.
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

# based on https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/ 2/08/19


# path to train csv
train_csv = "50_examples_final.csv"
test_csv = "test.csv"
images_folder = "train"

# Define number of epochs and batch size
NUM_EPOCHS = 50
BATCH_SIZE = 8

# Initialise number of training and test images
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

# Open train csv and get list of labels in train set and test set
f = pd.read_csv(train_csv)
trainLabels = f['landmark_id'].unique()


# Open test csv and get list of testLabels
f = pd.read_csv(test_csv)
testLabels = f['landmark_id'].unique()


# Build labelBinarizer for one-hot encoding labels and then encode the testing labels
lb = LabelBinarizer()
# fit using unique labels only
lb.fit(list(trainLabels))
# transform testLabels into binary one hot encoded tesLabels
testLabels = lb.transform(testLabels)

#sample image constructor for data augmentation, not currently in use
# performing data argumentation by training image generator
# randomly rotates flips shears etc
dataAugmentation = image.ImageDataGenerator(rotation_range=30, zoom_range=0.20,
                                           fill_mode="nearest", shear_range=0.20, horizontal_flip=True,
                                           width_shift_range=0.1, height_shift_range=0.1)

# initialsise training and testing image generators
# both methods currently do not apply aug but in the case this is applied  it will only be applied to the traingen
# both calls use mode train
trainGen = csv_image_generator(train_csv, images_folder, BATCH_SIZE, lb, mode="train", aug=dataAugmentation)
testGen = csv_image_generator(test_csv, images_folder, BATCH_SIZE, lb, mode="train")


stages = (3, 4, 6)
filters = (64, 128, 256, 256)

img_size = 224


# Measure performance using cross entropy. Always positive and equal to 0 if predicted == output.
# Want to minimise the cross-entropy by changing layer variables
# Cross-entropy function calculates softmax internally so use output of model(...) directly
# py_x = ResNet.ResNet.build(img_size, img_size, 3, num_classes, stages, filters)
model = ResNet.ResNet()
model = model.build(width=img_size, height=img_size, depth=3, classes=len(lb.classes_), stages=stages, filters=filters)

# model.summary()
# plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

# use of adam optimizer
# accuracy is a good metric for classification tasks
optimizer = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# train the network
# trainGen yields batches of data and labels
# steps per epoch must be supplied eother keras doesnt know when one epoch begins and another ends
# steps per epoch are caluclated from the number of images divided batch size
model_history = model.fit_generator(trainGen,
                                    steps_per_epoch=NUM_TRAIN_IMAGES//BATCH_SIZE,
                                    validation_data=testGen,
                                    validation_steps=NUM_TEST_IMAGES//BATCH_SIZE,
                                    epochs=NUM_EPOCHS)
#model_history = model.fit(images, labels, epochs=20, batch_size=batch_size)
model.save("ResNet_trained_model_50_lr_001")

# test the model on test dataset
# Evaluate the model
# reinitialise the testGen this time in eval  mode
TEST_BATCH_SIZE = BATCH_SIZE
testGen = csv_image_generator(test_csv, images_folder, TEST_BATCH_SIZE, lb, mode="eval", aug=None)

# make predicitions of test images finding the index of the label with the corresponding largest pred prob
predIdxs = model.predict_generator(testGen, steps=(NUM_TEST_IMAGES//TEST_BATCH_SIZE)+1)
predIdxs = np.argmax(predIdxs, axis=1)

# show classification report
print("~~~~~~~~~Evaluating network~~~~~~~~~~~")
print(predIdxs)
# report = classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_)
# report_dataframe = pd.DataFrame(report).transpose()
# report_dataframe.to_csv(r'resnet_report_temp.csv')






