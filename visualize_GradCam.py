# import the necessary packages
from GradCAM import GradCAM
from tensorflow.keras.applications import ResNet50
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

def image_name(image_path):
    regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).jpg"
    m = re.match(regex, image_path)
    return m.group(1) + "_" + m.group(2)

def get_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--cls_num', type=int, default=11)
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument("-i", "--images_dir", required=True,
                        help="path to the input image")
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-w", "--weights", type=str, default="imagenet",
                        help="model to be used")
    return parser.parse_args()


args = get_args()

# initialize the model to be VGG16
Model = VGG16
if args.weights == "imagenet":
    model = Model(weights="imagenet")
else:
    model = Model(weights=None, classes=args.cls_num)
    # network = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)
    # network.load_weights(args.weights).expect_partial()
    # network.assign_weights(model)
    model.load_weights(args.weights).expect_partial()



#load the pre-trained CNN from disk
print("[INFO] loading model...")
# network = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)
# # network.load_weights("C:\\Users\\lotan\\Downloads\\class_exp3\\checkpoint").expect_partial()
# network.assign_weights(model)
# # model = network.build_new_model()
# # model = network

def get_gradCam_image(image_path, output_path):

    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = np.array(Image.open(image_path))

    # resized = np.resize(orig, (224, 224))
    # load the input image from disk (in Keras/TensorFlow format) and
    # preprocess it
    image = load_img(image_path, target_size=args.input_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # use the network to make predictions on the input image and find
    # the class label index with the largest corresponding probability
    preds = model.predict(image)
    i = np.argmax(preds[0])
    # decode the ImageNet predictions to obtain the human-readable label
    # decoded = imagenet_utils.decode_predictions(preds)
    # (imagenetID, label, prob) = decoded[0][0]
    label = np.argmax(preds[0])
    prob = preds[0][i]
    label = "{}: {:.2f}%".format(label, prob * 100)
    print("[INFO] {}".format(label))

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # draw the predicted label on the output image
    # cv2.rectangle(title, (0, 0), (340, 40), (0, 0, 0), -1)
    # cv2.putText(title, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    # 	0.8, (255, 255, 255), 2)
    # display the original image and resulting heatmap and output image
    # to our screen
    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=700)

    plt.figure()
    plt.imshow(output)
    plt.title(label)
    plt.savefig(os.path.join(output_path, image_name(image_path)))

    # cv2.imshow("Output" + " " + label, output)
    # cv2.waitKey(0)

def get_results_for_imagesdir(input_path, output_path):
    dirs = os.listdir(input_path)
    for dir_name in dirs:
        dir_path = os.path.join(input_path, dir_name)
        output_dir_path = os.path.join(output_path, dir_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        images = os.listdir(dir_path)
        for image_name in images:
            full_path = os.path.join(dir_path, image_name)
            print(full_path)
            get_gradCam_image(full_path, output_dir_path)



get_results_for_imagesdir(args.images_dir, args.output_path)