import numpy as np

def pixel_accuracy(y_true, y_pred):
    """
    Compute the pixel accuracy.
    
    :param y_true: Ground truth image (2D array)
    :param y_pred: Predicted image (2D array)
    :return: Pixel accuracy as a percentage
    """
    # Ensure the input images are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate the total number of pixels
    N = y_true.size
    
    # Use the Kronecker delta function, which is 1 where the prediction and truth are equal
    correct_predictions = np.sum(y_true == y_pred)
    
    # Calculate the pixel accuracy
    accuracy = (correct_predictions / N) * 100
    return accuracy



def mean_iou(y_true, y_pred, num_classes):
    """
    Calculate the Mean Intersection over Union (mIoU).
    
    :param y_true: Ground truth labels (2D array)
    :param y_pred: Predicted labels (2D array)
    :param num_classes: Number of classes
    :return: Mean IoU value
    """
    iou_list = []
    for c in range(num_classes):
        true_positive = np.sum((y_true == c) & (y_pred == c))
        false_positive = np.sum((y_true != c) & (y_pred == c))
        false_negative = np.sum((y_true == c) & (y_pred != c))
        
        # Calculate the IoU for this class
        iou = true_positive / (true_positive + false_positive + false_negative + 1e-6)
        
        iou_list.append(iou)

    # Calculate the mean IoU across all classes
    miou = np.mean(iou_list)
    return miou
