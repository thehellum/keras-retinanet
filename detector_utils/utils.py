import filetype
import cv2
import matplotlib.pyplot as plt

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


def get_data_type(path):
    kind = filetype.guess(path)
    if kind is None:
        print('Cannot guess file type!')
        return

    print('File MIME type: %s' % kind.mime)
    return kind.mime.split("/")[0] 


def read_classes(csv_path):
    classes = {}
    file = open(csv_path, "r")
    for line in file:
        data = line[:-1].split(',')
        classes[int(data[1])] = data[0]

    return classes


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


def insert_detections(image, boxes, scores, labels, classes, score_threshold=0.5):
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < score_threshold:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(image, b, color=color)
        
        caption = "{} {:.3f}".format(classes[label], score)
        draw_caption(image, b, caption)

    return image


def display_image(image):    
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    