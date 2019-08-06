# Import keras
import keras

# Import keras_retinanet
from keras_retinanet import models
from keras_retinanet import backend

# Import miscellaneous modules
import cv2
import os
import numpy as np
import time
import argparse

# Set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# Self-made libraries/files
from detector_utils.eval import nms, nms_consider_label
from detector_utils.utils import get_data_type, read_classes, load_image, insert_detections, display_image



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



def main(args=None):
    nms_threshold = 0.3
    score_threshold = 0.3

    parser = argparse.ArgumentParser(description='SRC')
    parser.add_argument('src_path', help='Path to the input directory.')
    parser.add_argument('--weights', type=str, default=os.path.join('snapshots', 'resnet50_coco_best_v2.1.0.h5'),
                        help='Path for weights. Should be added to snapshots')
    parser.add_argument('--classes', type=str, default=None,
                        help='Path for csv file containing classes.')

    args = parser.parse_args()
    src_path = args.src_path


    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # Load model. Models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    model = models.load_model(args.weights, backbone_name='resnet50')

    try:
        classes = read_classes(args.classes)
    except TypeError:
        #classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        classes = {0: 'motor_vessel', 1: 'kayak', 2: 'sailboat_motor', 3: 'sailboat_sail'}

    # Determine filetype
    filetype = get_data_type(src_path)


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
            boxes /= scale

            # Non-max suppress more strictly
            boxes, scores, labels = nms(scores, boxes, labels, score_threshold, nms_threshold)
            print("Processing time: ", time.time() - start)

            frame = insert_detections(frame, boxes, scores, labels, classes, score_threshold)

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
        boxes /= scale

        # Non-max suppress more strictly
        boxes, scores, labels = nms(scores, boxes, labels, score_threshold, nms_threshold)
        print("processing time: ", time.time() - start)

        # Display detections
        image = insert_detections(draw, boxes, scores, labels, classes, score_threshold)
        display_image(image)


    else:
        print("ERROR in opening file at spesified path")
        return




if __name__ == '__main__':
    main()
