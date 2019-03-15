import random
import numpy as np
import pickle
import pandas as pd
from keras.utils import np_utils

NUM_CLASSES = 7
IMG_SIZE = 48
TRAIN_END = 28708
TEST_START = TRAIN_END + 1

class DataTransformer(object):


    @staticmethod
    def manual_balance_batches(Y):
        """
        Format used to balance the batches - researched based approach to ensure that an unbalanced dataset is balanced.
        :param Y: Labels to balance.
        :type Y: Numpy Array of Labels.
        :return: Dictionary containing all the label index's
        :rtype: Dictionary
        """
        # Create dictionary to store the sorted labelled lists.
        batch_dict = {}

        "Itterate over 36 batches (900 / 25)"
        for u in range(36):
            batch = []
            "Over each emotional range add three random values from each to the batch."
            for i in range(0, 7):
                for j in range(3):
                    indices = [k for k, x in enumerate(Y) if x == i]
                    rand_indices = random.choice(indices)
                    "Ensure the same random indicies isnt alrady in the batch"
                    if not batch.__contains__(rand_indices):
                        batch.append(rand_indices)
                    else:
                        while batch.__contains__(rand_indices):
                            rand_indices = random.choice(indices)

                        batch.append(rand_indices)

            for h in range(0, 4):
                indices = [l for l, p in enumerate(Y) if p == h]
                rand_indices = random.choice(indices)
                if not batch.__contains__(rand_indices):
                    batch.append(rand_indices)
                else:
                    while batch.__contains__(rand_indices):
                        rand_indices = random.choice(indices)

                    batch.append(rand_indices)

            batch_dict[u] = batch

        return batch_dict

    @staticmethod
    def manual_convert_ohe(y):
        """
        Converts the labels into One Hot Encoded labels.
        :param y: labels
        :type y: NumpyArray
        :return: OHE Label array.
        :rtype: Array
        """
        N = len(y)
        K = 7
        ind = np.zeros((N, K))
        for i in range(N):
            ind[i, y[i]] = 1
        return ind

    @staticmethod
    def split_data(list):
        train = list[0:TRAIN_END]
        test = list[TEST_START:]
        return train, test

    @staticmethod
    def duplicate_input_layer(array_input, size):
        vg_input = np.empty([size, 48, 48, 3])
        for index, item in enumerate(vg_input):
            item[:, :, 0] = array_input[index]
            item[:, :, 1] = array_input[index]
            item[:, :, 2] = array_input[index]
        return vg_input

    def convert_data(self, data):
        """
        Converts all training, testing and validation datasets into the same format.
        """
        training_x = np.asarray(data[0])
        training_x = training_x.astype(np.float32)
        training_oh_y = self.manual_convert_ohe(data[1]).astype(np.float32)

        validation_x = np.asarray(data[2])
        validation_x = validation_x.astype(np.float32)
        validation_oh_y = self.manual_convert_ohe(data[3]).astype(np.float32)

        testing_x = np.asarray(data[4])
        testing_x = testing_x.astype(np.float32)
        testing_oh_y = self.manual_convert_ohe(data[5]).astype(np.float32)

        return {"training": [training_x, training_oh_y], "testing": [testing_x, testing_oh_y], "validation": [validation_x, validation_oh_y]}

    def transform_csv_to_img(self, loaded_data):

        img = loaded_data["pixels"][0]  # first image
        val = img.split(" ")

        x_pixels = np.array(val, 'float32')
        x_pixels /= 255
        x_reshaped = x_pixels.reshape(48, 48)

        return x_reshaped

    def pandas_vector_to_list(self, pandas_df):
        py_list = [item[0] for item in pandas_df.values.tolist()]
        return py_list

    def process_emotion(self, emotion):
        """
        Takes in a vector of emotions and outputs a list of emotions as one-hot vectors.
        :param emotion: vector of ints (0-7)
        :return: list of one-hot vectors (array of 7)
        """
        emotion_as_list = self.pandas_vector_to_list(emotion)
        y_data = []
        for index in range(len(emotion_as_list)):
            y_data.append(emotion_as_list[index])

        # Y data
        y_data_categorical = np_utils.to_categorical(y_data, NUM_CLASSES)
        return y_data_categorical

    def process_pixels(self, pixels, img_size=IMG_SIZE):
        """
        Takes in a string (pixels) that has space separated ints. Will transform the ints
        to a 48x48 matrix of floats(/255).
        :param pixels: string with space separated ints
        :param img_size: image size
        :return: array of 48x48 matrices
        """

        pixels_as_list = self.pandas_vector_to_list(pixels)

        np_image_array = []
        for index, item in enumerate(pixels_as_list):
            # 48x48
            data = np.zeros((img_size, img_size), dtype=np.uint8)
            # split space separated ints
            pixel_data = item.split()

            # 0 -> 47, loop through the rows
            for i in range(0, img_size):
                # (0 = 0), (1 = 47), (2 = 94), ...
                pixel_index = i * img_size
                # (0 = [0:47]), (1 = [47: 94]), (2 = [94, 141]), ...
                data[i] = pixel_data[pixel_index:pixel_index + img_size]

            np_image_array.append(np.array(data))

        np_image_array = np.array(np_image_array)
        # convert to float and divide by 255
        np_image_array = np_image_array.astype('float32') / 255.0
        return np_image_array

class ImageLoader(object):

    def load_csv_data(self, file_path):
        raw_data_csv_file_name = file_path
        raw_data = pd.read_csv(raw_data_csv_file_name)
        return raw_data

    @staticmethod
    def import_images_pickle(file_name):
        """
       Loads data from a pickle file.
       :param file_name: The name of the file to open.
       :type file_name: String
       :return: True or False as to whether file has been opened.
       :rtype: Boolean
       """
        pickle_file = file_name

        with open(pickle_file, 'rb') as f:
            load = pickle.load(f)
        try:
            training_x, training_y = load[0], load[1]
            validation_x, validation_y = load[2], load[3]
            testing_x, testing_y = load[4], load[5]
        except FileNotFoundError:
            raise FileNotFoundError

        return [training_x, training_y, validation_x, validation_y, testing_x, testing_y]
