from sklearn.metrics import precision_recall_curve
import numpy as np


def find_threshold_f1(trues, logits, eps=1e-9):
    if len(trues.shape) > 1:
        threshold = []
        for i in range(trues.shape[1]):
            precision, recall, thresholds = precision_recall_curve(trues[:,i], logits[:,i])
            f1_scores = 2 * precision * recall / (precision + recall + eps)
            threshold.append(float(thresholds[np.argmax(f1_scores)]))
        return threshold
    else:
        precision, recall, thresholds = precision_recall_curve(trues, logits)
        f1_scores = 2 * precision * recall / (precision + recall + eps)
        threshold.append(float(thresholds[np.argmax(f1_scores)]))
        return threshold