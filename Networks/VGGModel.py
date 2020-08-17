from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

import tensorflow as tf


class VGGModel(NNInterface):
    def __init__(self, classes_num, input_size):
        super().__init__(classes_num, input_size)
        self.output_path = None
        self.input_size = input_size
        vgg_conv = vgg16.VGG16(weights=None, include_top=False, classes=classes_num, input_shape=(input_size[0], input_size[0], 3))

        self.last_layer = tf.keras.layers.Dense(classes_num, activation='softmax')
        vgg_conv.summary()

        # for layer in vgg_conv.layers[:]:
        #     layer.trainable = False

        self.__model = tf.keras.Sequential()
        self.__model.add(vgg_conv)
        self.__model.add(tf.keras.layers.Flatten())
        self.__model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.__model.add(tf.keras.layers.Dropout(0.5))
        self.__model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.__model.add(tf.keras.layers.Dropout(0.5))

        self.__model.add(self.last_layer)

        self.__model.summary()

        # if os.path.exists(self.get_last_ckpt_path()):
        #     self.load_model(self.get_last_ckpt_path())
        #     print("loads last weights")

    def assign_weights(self, vgg_model):
        j = 0
        for i, layer in enumerate(self.__model.layers):
            if i == 0:
                for sub_layer in layer.layers[:]:
                    print("assign {} to {}".format(vgg_model.layers[j].name, sub_layer.name))
                    vgg_model.layers[j].set_weights(sub_layer.get_weights())
                    j += 1

                    # model.add(sub_layer)
                    # # print(sub_layer.name)
                    # x = sub_layer(x)
            else:
                if layer.name.startswith("dropout"):
                    continue
                print("assign {} to {}".format(vgg_model.layers[j].name, layer.name))

                vgg_model.layers[j].set_weights(layer.get_weights())
                j += 1

                # model.add(layer)



    def build_new_model(self):
        inputs = tf.keras.Input(shape=(self.input_size[0], self.input_size[0], 3))
        model = tf.keras.Sequential(name="gradCam")
        # inputs = tf.keras.layers.Input(shape=(self.input_size[0], self.input_size[1], 3))
        model.add(inputs)
        # inputs = self.__model.layers[0].layers[0].input
        # x = inputs
        for i, layer in enumerate(self.__model.layers):
            if i == 0:
                for sub_layer in layer.layers[1:]:
                    model.add(sub_layer)
                    # print(sub_layer.name)
                    # x = sub_layer(x)
            else:
                model.add(layer)

                # x = layer(x)
        # model = tf.keras.Model(inputs=inputs, outputs=x)
        # model.build(input_shape=(None, self.input_size[0], self.input_size[1], 3) )
        return model


    def get_last_conv_layer(self):
        return self.last_conv_layer

    def get_last_layer(self):
        return self.last_layer


    def build_fill_vgg_model(self, classes_num, input_size):
        model = Sequential()
        model.add(Conv2D(input_shape=(input_size[0], input_size[1], 3), filters=64, kernel_size=(3, 3), padding="same",
                         activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))

        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))

        model.add(Dense(units=classes_num, activation="softmax"))

    def update_output_path(self, output_path):
        self.output_path = output_path

    def get_model_object(self):
        return self.__model

    def call(self, x, training=False, **kwargs):
        x = vgg16.preprocess_input(x)
        return self.__model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)

    def freeze_status(self):
        for i, layer in enumerate(self.__model.layers):
            # if i == 0:
            #     for sub_layer in layer.layers[:]:
            #         print("layer {} is trainable {}".format(sub_layer.name, sub_layer.trainable))
            # else:
            print("layer {} is trainable {}".format(layer.name, layer.trainable))

    def save_model(self, iter_num, output_path):
        output_path = os.path.join(output_path, "ckpts")
        checkpoint_path = "weights_after_{}_iterations".format(iter_num)
        self.__model.save_weights(os.path.join(output_path, checkpoint_path))

    def load_model(self, ckpt_path):
        self.__model.load_weights(ckpt_path)

    def get_last_ckpt_path(self):
        output_path = os.getcwd() if self.output_path is None else self.output_path
        output_path = os.path.join(output_path, "last_ckpts")
        return os.path.join(output_path, "ckpt")
