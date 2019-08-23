import pandas as pd
from keras.preprocessing import image
from keras import optimizers
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import numpy as np

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


model = load_model('')
testCSV = "test.csv"
images_folder = "/test"
TEST_BATCH_SIZE = 16


f = pd.read_csv(testCSV)
testLabels = f['landmark_id'].unique()
lb = LabelBinarizer()
lb.transform(testLabels)

testGen = csv_image_generator(testCSV, images_folder, TEST_BATCH_SIZE, lb, mode="eval", aug=None)

evals = evaluate_generator(testGen, max_queue_size=10, workers=20, use_multiprocessing=True, verbose=1)
print(model.metrics_names)
print(evals)

