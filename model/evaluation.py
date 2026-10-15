def generate_conclusion(deepfake_score, threshold):
    """
    Interpreting the model output conclusion.
    The purpose is to generate a conclusion of deepfake or not.
    """

    prediction = (deepfake_score >= threshold)

    return {
        "prediction": prediction,
        "deepfake_score": deepfake_score,
        "conclusion": (
            "Predicted deepfake" if prediction == 1 else "Predicted real"
        ),
    }

def calculate_metrics(predictions, true_labels):
    """
    Measure performance across different samples.
    Core metrics such as:
    - Accuracy
    - Prediction
    - Recall
    - F1
    - ROC-AUC
    - Confusion Matrix
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for predicted, actual in zip(predictions, true_labels):
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else None #avoiding div by 0
    recall = tp / (tp + fn) if tp + fn else None #avoiding div by 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None #avoiding div by 0

    return {
        "sample_count": len(predictions),
        "accuracy": (tp + tn) / len(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        #Rows: actual real/deepfake; columns: predicted real/deepfake.
        #roc-auc should be added also
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
