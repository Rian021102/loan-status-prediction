import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
def eda(X_train, y_train):
    #print shape of X_train and X_test
    print('X_train shape: ',X_train.shape)

    #print information of X_train
    print('X_train data types: ',X_train.info())

    #print describe of X_train in table
    print('X_train describe: ',X_train.describe())

    
    #print skewness of X_train
    print('X_train skewness: ',X_train.skew())
    
    #check missing value
    print('X_train missing value: ',X_train.isnull().sum())

    #check duplicate value
    print('X_train duplicate value: ',X_train.duplicated().sum())
    
    #group all numerical variables
    num=X_train.select_dtypes(include=['int64','float64'])
    print('Numerical variables: ',num.columns)

    #group all categorical variables
    cat=X_train.select_dtypes(include=['object'])
    print('Categorical variables: ',cat.columns)

    #boxplot numerical variables with figure size 20,10
    plt.figure(figsize=(25,20))
    sns.boxplot(data=num)

   #remove outliers using IsolationForest
    iso=IsolationForest(contamination=0.1)
    yhat=iso.fit_predict(num)
    mask=yhat!=1
    X_train=X_train[mask]
    y_train=y_train[mask]
    
    
    #check shape of X_train and y_train
    print('X_train shape after removing outliers: ',X_train.shape)
    print('y_train shape after removing outliers: ',y_train.shape)

    
    #plot pie chart for loan_condition
    plt.figure(figsize=(10,10))
    plt.pie(y_train.value_counts(),labels=['Good Loan','Bad Loan'],autopct='%1.1f%%',shadow=True)

    #plot bar chart for loan_condition
    plt.figure(figsize=(10,10))
    sns.countplot(data=X_train, x='loan_condition')

    #plot histogram for X_train
    X_train.hist(figsize=(20,20))


    #filter with loan_condition is bad loan
    bad_loan=X_train[X_train['loan_condition']=='Bad Loan']

    #create function of countplot for bad_loan
    def countplot_bad_loan(col):
        plt.figure(figsize=(25,15))
        sns.countplot(data=bad_loan,x=col, order=bad_loan[col].value_counts().index)
        plt.show()

    #plot countplot for home_ownership using function
    countplot_bad_loan('home_ownership')

    #plot countplot for income_category using function
    countplot_bad_loan('income_category')

    #plot countplot for term using function
    countplot_bad_loan('term')

    #plot countplot for purpose using function
    countplot_bad_loan('purpose')

    #plot countplot for interest_payments using function
    countplot_bad_loan('interest_payments')

    #plot countplot for application_type using function
    countplot_bad_loan('application_type')

    #plot countplot for grade using function
    countplot_bad_loan('grade')

    #plot countplot for region using function
    countplot_bad_loan('region')

    #application type with loan condition
    plt.figure(figsize=(10,10))
    sns.countplot(data=X_train, x='application_type',hue='loan_condition')
    plt.show()

    #countplot income category with hue region
    plt.figure(figsize=(10,10))
    sns.countplot(data=X_train, x='income_category',hue='region')
    plt.show()

    #plot histogram for bad_loan
    bad_loan.hist(figsize=(20,20))

    #plot pairplot for bad_loan with figure size 20,20
    plt.figure(figsize=(25,25))
    sns.pairplot(data=bad_loan)
    plt.show()

    #plot correlation heatmap for X_train
    plt.figure(figsize=(25,25))
    sns.heatmap(X_train.corr(),annot=True)
    plt.show()
    


    return X_train, y_train





