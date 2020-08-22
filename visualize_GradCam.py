# import the necessary packages
from GradCAM import GradCAM
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import nn_builder
import os
import re
from losses import FeaturesLoss

def image_name(image_path):
    regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).(jpg|JPEG)"
    m = re.match(regex, image_path)
    return m.group(1) + "_" + m.group(2)

def get_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="VGGModel", choices=["imagenet", "amazon", "DOC"])
    parser.add_argument('--cls_num', type=int, default=11)
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument("-i", "--images_dir", required=True,
                        help="path to the input image")
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-w", "--weights", type=str, default="imagenet",
                        help="model to be used")
    parser.add_argument("-l", "--label_map", default=None)


    parser.add_argument("-t", "--templates_dir", required=True,
                        help="path to the templates images")
    return parser.parse_args()


def get_model(type, cls_num, weights_path, templates_path):
    print("[INFO] loading model...")

    # initialize the model to be VGG16
    Model = VGG16
    if type == "imagenet":
        model = Model(weights="imagenet")
        # loss = tf.keras.losses.CategoricalCrossentropy()
        # prediction_layer = "predictions"

    elif type == "amazon":
        model = Model(weights=None, classes=cls_num)
        network = nn_builder.get_network("VGGModel", cls_num, args.input_size)
        network.load_weights(weights_path).expect_partial()
        network.assign_weights(model)
        # loss = tf.keras.losses.CategoricalCrossentropy()
        # prediction_layer = "predictions"
    else:
        model = Model(weights=None, classes=cls_num)
        model.load_weights(weights_path).expect_partial()
        # templates_paths = os.listdir(templates_path)
        # templates = np.concatenate(
        #     [read_image(os.path.join(templates_path, path), args.input_size) for path in templates_paths])
        #
        # loss = FeaturesLoss(templates, model)
        # prediction_layer = "fc2"

    return model


def read_image(image_path, input_size):
    image = load_img(image_path, target_size=input_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image




args = get_args()

def read_label_map(file):
    with open(file, 'r') as f:
        label_map = dict()
        lines = f.read().splitlines()
        for line in lines:
            line = line.replace("{", "").replace("}", "").replace(",", "")
            line_args = line.split(":")
            label_map[line_args[0].strip()] = line_args[1].strip()
        return label_map





def get_gradCam_image(model, cls_num, image_path, output_path):

    if args.label_map is not None:
        label_map = read_label_map(args.label_map)
    else:
        label_map = None

    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = np.array(Image.open(image_path).convert('RGB'))

    # resized = np.resize(orig, (224, 224))
    # load the input image from disk (in Keras/TensorFlow format) and
    # preprocess it
    # image = load_img(image_path, target_size=args.input_size)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    image = read_image(image_path, args.input_size)

    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict(image)
    # decode the ImageNet predictions to obtain the human-readable label
    # decoded = imagenet_utils.decode_predictions(preds)
    # (imagenetID, label, prob) = decoded[0][0]
    cam = GradCAM(model, cls_num)

    all_outputs = []
    labels_str = ""
    max_labels_idx = np.argsort(preds[0])
    for i in range(3):
        print(i)
        idx = max_labels_idx[args.cls_num-1 - i]
        prob = preds[0][idx]
        label = str(idx)
        if label_map is not None:
            label = "{}, {}".format(label_map[label], label)
        labels_str += "{}: {:.2f}%\n".format(label, prob * 100)

        heatmap = cam.compute_heatmap(np.copy(image), idx)
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)



        all_outputs.append(np.hstack([orig, heatmap, output]))

    i = np.argmax(preds[0])
    label = str(np.argmax(preds[0]))
    prob = preds[0][i]
    if label_map is not None:
        label = "{}, {}".format(label_map[label], label)
    label = "{}: {:.2f}%".format(label, prob * 100)
    print("[INFO] {}".format(label))

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, cls_num)
    heatmap = cam.compute_heatmap(image, i)
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))


    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)







    output = np.hstack([orig, heatmap, output])
    output = imutils.resize(output, height=700)

    output = np.vstack(all_outputs)
    output = imutils.resize(output, height=2100)




    fig = plt.figure()
    plt.imshow(output)
    plt.title(labels_str)
    plt.savefig(os.path.join(output_path, image_name(image_path)), bbox_inches='tight')
    plt.close(fig)

    # cv2.imshow("Output" + " " + label, output)
    # cv2.waitKey(0)

def get_results_for_imagesdir(weights_type, input_path, output_path, templates_path):
    ## read templates
    if weights_type == "DOC":
        templates_paths = os.listdir(templates_path)
        templates = np.concatenate([read_image(os.path.join(templates_path, path), args.input_size) for path in templates_paths])


    dirs = os.listdir(input_path)
    model = get_model(weights_type, args.cls_num, args.weights, templates_path)
    for dir_name in dirs:
        dir_path = os.path.join(input_path, dir_name)
        output_dir_path = os.path.join(output_path, dir_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        images = os.listdir(dir_path)
        for image_name in images:
            full_path = os.path.join(dir_path, image_name)
            get_gradCam_image(model, args.cls_num, full_path, output_dir_path)



get_results_for_imagesdir(args.type, args.images_dir, args.output_path, args.templates_dir)