import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def xgb_model(X_train_res,y_train_res,X_test_res,y_test_res):

    model=xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42)

    model.fit(X_train_res,y_train_res)

    y_pred=model.predict(X_test_res)

    y_pred_prob=model.predict_proba(X_test_res)[:,1]

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

    #print feature importance
    print('Feature importance: ',model.feature_importances_)

    #plot feature importance
    plt.figure(figsize=(10,10))
    plt.barh(X_train_res.columns,model.feature_importances_)
    plt.show()

    return model

