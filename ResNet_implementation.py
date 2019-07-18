import pandas as pd
import tensorflow as tf
import pathlib
import os
import ResNet

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
reduced_csv = "20_examples.csv/"
images_folder = "train/"
copy_chosen_images(reduced_csv, images_folder)

# split data into train and test set
# Measure performance using cross entropy. Alwyas positive and equal to 0 if predicted == output.
# Want to minimise the cross-entropy by changing layer variables
# Cross-entropy function calculates softmax internally so use output of model(...) directly
py_x = ResNet
# Define cross entropy for each image
Y_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=py_x)

# Take average of cross entropy for all classified images to find single scalar value to optimise network variables
cost = tf.reduce_mean(Y_)

# Use optimiser to minimise the above cost.
# Using RMSProp algorithm. Algo also divides the learning rate by exponentially decaying average of squared gradients
# Suggested decay param 0.9, learning rate 0.001
optimiser = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# define predict_opm, the index with the largest value across dimensions from the output
predict_op = tf.argmax(py_x, 1)

# Define networks running section
# Perform in batches

# implement Tensorflow session
with tf.Session() as sesh:
    tf.global_variables_initializer().run()
    for i in range(100):
        # Get small batch of training examples that holds images and corresponding labels
        training_batch = zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX)+1, batch_size))

        # Put batch into a feed_dict with appropriate names for placeholders in the graph
        # Run optimiser using this batch of training data
        for start, end in training_batch:
            sesh.run(optimiser, feed_dict={X: trainX[start:end],
                                           Y: trainY[start:end],
                                           p_keep_conv: 0.8,
                                           p_keep_hidden: 0.5})

        # At the same time get a shuffled batch of test samples
        test_indices = np.arange(len(testX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        # For each iteration, display the accuracy of the batch
        print(i, np.mean(np.argmax(testY[test_indices], axis=1) == sesh.run(predict_op,
                                                                            feed_dict={X: testX[test_indices],
                                                                                       Y: testY[test_indices],
                                                                                       p_keep_conv: 1.0,
                                                                                       p_keep_hidden: 1.0})))





