import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)

RFC_FOREST_DEPTH = 2

def model_train(X_train, y_train):
   model = RandomForestClassifier(n_estimators=5, max_depth=RFC_FOREST_DEPTH, random_state=1993)
   model.fit(X=X_train, y=y_train)
   return model

def model_evaluate(model, X_test, y_test, float_precision=4):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred,)
    precision = precision_score(y_true=y_test, y_pred=y_pred,)
    recall = recall_score(y_true=y_test, y_pred=y_pred,)
    f1 = f1_score(y_true=y_test, y_pred=y_pred,)

    metrics = {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
     }

    return json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precision,),
    )