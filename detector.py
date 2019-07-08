# Import keras
import keras

# Import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# Import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import sys
import argparse

# Set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# Self-made libraries/files
import extract_filetype as file


"""
Variables
""" 

# Load label to names mapping for visualization purposes (COCO)
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
precision = 0.3

"""
Functions
"""

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def load_retinanet():
    # Adjust this to point to your downloaded/trained model
    # Models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    model_path = os.path.join('snapshots', 'resnet50_coco_inference_v2.1.0.h5')

    # Load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    # If the model is not converted to an inference model, use the line below
    # See: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    # model = models.convert_model(model)

    # print(model.summary())

    return model


def arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='SRC')
    parser.add_argument('src_path', help='Path to the input directory.')

    parsed_args = parser.parse_args(sys.argv[1:])

    # Check if spesified input exist
    if os.path.exists(parsed_args.src_path):
        print("Src path exist at: ", parsed_args.src_path)
    else:
        print("Src does not exist at specified path")

    return parsed_args.src_path


def load_image(image_path):
    # Load image
    image = read_image_bgr(image_path)

    # Copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    return image, draw, scale


def insert_detections(image, boxes, scores, labels, precision=0.5):
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < precision:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(image, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image, b, caption)

    return image


def display_image(image):    
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    



def main(args=None):
    """ Init model """
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # Load model
    model = load_retinanet()

    # Parse args
    src_path = arg_parser()

     # Determine filetype
    filetype = file.get_type(src_path)


    if filetype == "video":
        cap = cv2.VideoCapture(src_path)

        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()

            # Preprocess image for network
            image = preprocess_image(frame)
            image, scale = resize_image(image)

            # Process image using model    
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            print("Processing time: ", time.time() - start)

            # Correct for image scale
            boxes /= scale

            frame = insert_detections(frame, boxes, scores, labels, precision)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


    elif filetype == "image":
        image, draw, scale = load_image(src_path)

        # Process image using model    
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # Correct for image scale
        boxes /= scale

        # Display detections
        image = insert_detections(draw, boxes, scores, labels, precision)
        display_image(image)


    else:
        print("ERROR in opening file at spesified path")
        return




if __name__ == '__main__':
    main()