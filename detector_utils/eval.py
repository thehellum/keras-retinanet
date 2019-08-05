import numpy as np

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_prediction = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    intersection_x = min(gt_box[2],prediction_box[2]) - max(gt_box[0],prediction_box[0])
    intersection_y = min(gt_box[3],prediction_box[3]) - max(gt_box[1],prediction_box[1])
    if intersection_x > 0 and intersection_y > 0:
        intersection = intersection_x * intersection_y
        union = area_gt + area_prediction - intersection
        iou = (float(intersection) / union)
    else:
        iou = 0
    return iou



def nms_consider_label(scores, boxes, labels, score_threshold, nms_threshold):
    # Keep predictions where scores > threshold
    try:
        indices = np.concatenate(np.argwhere(scores[0] > score_threshold))
        scores = np.expand_dims(scores[0][indices], axis=0)
        boxes = np.expand_dims(boxes[0][indices], axis=0)
        labels = np.expand_dims(labels[0][indices], axis=0)
        
        # Suppress
        i = 1
        deleted = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            j = i
            while j < len(scores[0]):
                if (label == labels[0][j]):
                    if j not in deleted: 
                        if calculate_iou(boxes[0][j], box) >= nms_threshold:
                            indices = np.delete(indices, j-len(deleted))
                            deleted.append(j)
                j += 1
            i += 1

        scores = np.expand_dims(scores[0][indices], axis=0)
        boxes = np.expand_dims(boxes[0][indices], axis=0)
        labels = np.expand_dims(labels[0][indices], axis=0)
    
    except ValueError:
        scores = [[]]
        boxes = [[]]
        labels = [[]]

    return boxes, scores, labels



def nms(scores, boxes, labels, score_threshold, nms_threshold):
    # Only keep predictions where scores > threshold
    try:
        indices = np.concatenate(np.argwhere(scores[0] > score_threshold))
        scores = np.expand_dims(scores[0][indices], axis=0)
        boxes = np.expand_dims(boxes[0][indices], axis=0)
        labels = np.expand_dims(labels[0][indices], axis=0)

        # Suppress
        i = 1
        deleted = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            j = i
            while j < len(scores[0]):
                if j not in deleted: 
                    if calculate_iou(boxes[0][j], box) >= nms_threshold:
                        indices = np.delete(indices, j-len(deleted))
                        deleted.append(j)
                j += 1
            i += 1

        scores = np.expand_dims(scores[0][indices], axis=0)
        boxes = np.expand_dims(boxes[0][indices], axis=0)
        labels = np.expand_dims(labels[0][indices], axis=0)
    
    except ValueError:
        scores = [[]]
        boxes = [[]]
        labels = [[]]

    return boxes, scores, labels