# import random forest classifier, decision tree classifier, xgboost classifier, adaboost classifier, gradient boosting classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def modeling(X_train_rus, y_train_rus, X_test_num, y_test):
    models=[RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),DecisionTreeClassifier(),XGBClassifier()]
    model_names=['Random Forest Classifier','AdaBoost Classifier','Gradient Boosting Classifier','Decision Tree Classifier','XGB Classifier']
    accuracy=[]
    #create for loop for models and model_names and print accuracy score,confusion matrix and classification report for each model
    for model in range(len(models)):
        clf=models[model]
        clf.fit(X_train_rus,y_train_rus)
        y_pred=clf.predict(X_test_num)
        accuracy.append(accuracy_score(y_test,y_pred))
        print(model_names[model])
        print('Accuracy Score: ',accuracy_score(y_test,y_pred))
        print('Confusion Matrix: ',confusion_matrix(y_test,y_pred))
        print('Classification Report: ',classification_report(y_test,y_pred))
        print('\n')
    #create dataframe of accuracy and model_names
    accuracy_df=pd.DataFrame({'Accuracy':accuracy,'Model':model_names})
    #plot bar chart of accuracy_df
    plt.figure(figsize=(10,10))
    sns.barplot(data=accuracy_df,x='Accuracy',y='Model')
    plt.title('Accuracy of Models')
    plt.show()

    return models