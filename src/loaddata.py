import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(pathfile):
    df=pd.read_csv(pathfile)
    print(df.head())
    print(df.columns)
    #drop id
    df=df.drop(['id'],axis=1)
    #reset index
    df=df.reset_index(drop=True)

    #convert issue_d to datetime
    df['issue_d']=pd.to_datetime(df['issue_d'])

    #convert final_d to datetime
    df['final_d'] = pd.to_datetime(df['final_d'], format='%m%d%Y')
   
    #set x and y
    X=df.drop(['loan_condition_cat'],axis=1)
    y=df['loan_condition_cat']
    #split data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    return X_train,X_test,y_train,y_test