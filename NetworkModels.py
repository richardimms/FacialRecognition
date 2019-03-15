from keras import Sequential, Input, Model
from keras.applications import VGG16
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.python.saved_model import tag_constants, signature_constants
from keras import backend as K
from MachineLearningUtils import DataTransformer
import numpy as np

NUM_CLASSES = 7
IMG_SIZE = 48

class VGGModel():

    def __init__(self):
        K.set_learning_phase(0)
        self.vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')


    def get_vgg16_output(self, array_input, n_feature_maps):
        vg_input = DataTransformer.duplicate_input_layer(array_input, n_feature_maps)

        picture_train_features = self.vgg16.predict(vg_input)
        del (vg_input)

        feature_map = np.empty([n_feature_maps, 512])
        for idx_pic, picture in enumerate(picture_train_features):
            feature_map[idx_pic] = picture
        return feature_map

    def train_base_model(self,x_train_feature_map, x_test_feature_map, y_train, y_test ):
        """
        The function here is used to train the initial base model that will eventually be merged with the
        larger model to utilise transfer learning, here we have defined it as VGG16.
        :param training_dictionary:
        :return:
        """
        # build and train model
        top_layer_model = Sequential()
        top_layer_model.add(Dense(256, input_shape=(512,), activation='relu'))
        top_layer_model.add(Dense(256, input_shape=(256,), activation='relu'))
        top_layer_model.add(Dropout(0.5))
        top_layer_model.add(Dense(128, input_shape=(256,)))
        top_layer_model.add(Dense(NUM_CLASSES, activation='softmax'))

        adamax = Adamax()

        top_layer_model.compile(loss='categorical_crossentropy',
                                optimizer=adamax, metrics=['accuracy'])

        # train
        top_layer_model.fit(x_train_feature_map, y_train,
                            validation_data=(x_train_feature_map, y_train),
                            nb_epoch=100, batch_size=25)
        # Evaluate
        score = top_layer_model.evaluate(x_test_feature_map,
                                         y_test, batch_size=25)

        print("After top_layer_model training (test set): {}".format(score))

        return {"Model" : [self.vgg16, top_layer_model, adamax], "Training": [x_train_feature_map, y_train],
                "Testing": [x_test_feature_map, y_test]}

    def merge_models(self, initial_training_dict):

        vgg16, top_layer_model, adamax = initial_training_dict['Model']

        x_train_input, y_train = initial_training_dict['Training']
        x_test_input, y_test = initial_training_dict['Testing']

        # Merge two models and create the final_model_final_final
        inputs = Input(shape=(48, 48, 3))
        vg_output = vgg16(inputs)

        print("vg_output: {}".format(vg_output.shape))

        model_predictions = top_layer_model(vg_output)

        final_model = Model(input=inputs, output=model_predictions)
        final_model.compile(loss='categorical_crossentropy',
                            optimizer=adamax, metrics=['accuracy'])

        final_model_score = final_model.evaluate(x_train_input,
                                                 y_train, batch_size=25)

        print("Sanity check - final_model (train score): {}".format(final_model_score))

        final_model_score = final_model.evaluate(x_test_input,
                                                 y_test, batch_size=25)
        print("Sanity check - final_model (test score): {}".format(final_model_score))

        return final_model

    def save_model(self, model_to_save):
        print("Model input name: {}".format(model_to_save.input))
        print("Model output name: {}".format(model_to_save.output))

        # Save Model
        builder = saved_model_builder.SavedModelBuilder('./')
        signature = predict_signature_def(inputs={'images': model_to_save.input},
                                          outputs={'scores': model_to_save.output})
        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()

