import json

import pandas as pd
from sklearn.model_selection import train_test_split

from metrics_and_plots import plot_confusion_matrix, save_metrics
from model import model_evaluate, model_train
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN

def load_data(file_path):
    data = pd.read_csv(filepath_or_buffer=file_path,)
    x = data.drop(labels=TARGET_COLUMN, axis=1,)
    y = data[TARGET_COLUMN]
    return x, y,

def main():
    X, y = load_data(file_path=PROCESSED_DATASET)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model = model_train(X_train=X_train, y_train=y_train)

    metrics = model_evaluate(model=model, X_test=X_test, y_test=y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    # Save metrics into json file
    save_metrics(metrics=metrics)
    plot_confusion_matrix(model=model, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()