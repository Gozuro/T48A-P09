from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred, average='macro'):
    pass
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return precision, recall, f1
