import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test, cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")
    plt.close()

def save_metrics(metrics):
    with open(file="metrics.json", mode="w") as fp:
        json.dump(metrics, fp=fp)