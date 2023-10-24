#import logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def lr_model(X_train_res,y_train_res,X_test_res,y_test_res):
    model_lr=LogisticRegression(random_state=42)
    model_lr.fit(X_train_res,y_train_res)
    y_pred=model_lr.predict(X_test_res)
    y_pred_prob=model_lr.predict_proba(X_test_res)[:,1]
    print('Accuracy score: ',accuracy_score(y_test_res,y_pred))
    print('Confusion matrix: ',confusion_matrix(y_test_res,y_pred))
    print('Classification report: ',classification_report(y_test_res,y_pred))
    print('ROC AUC score: ',roc_auc_score(y_test_res,y_pred_prob))
    fpr,tpr,thresholds=roc_curve(y_test_res,y_pred_prob)
    plt.figure(figsize=(10,10))
    plt.plot(fpr,tpr,linewidth=2)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    return model_lr