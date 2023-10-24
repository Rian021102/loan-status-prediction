import pandas as pd
import numpy as np
#import random under sampling
from imblearn.under_sampling import RandomUnderSampler
#import label encoder
from sklearn.preprocessing import LabelEncoder

def process_train(X_train, y_train):

    X_train=X_train[['emp_length_int','annual_inc',
                        'loan_amount','interest_rate','dti',
                        'total_rec_prncp','recoveries','home_ownership','income_category',
                        'term','application_type','purpose','interest_payments',
                        'grade','region']]


    categorical=X_train[['home_ownership','income_category',
                        'term','application_type','purpose','interest_payments',
                        'grade','region']]


    #create label encoder categorical data on X_train
    le=LabelEncoder()
    X_train[categorical.columns]=X_train[categorical.columns].apply(lambda col: le.fit_transform(col))

    #perform random under sampling
    rus=RandomUnderSampler(random_state=42)
    X_train_res,y_train_res=rus.fit_resample(X_train,y_train)

    print("Data train processed")

    return X_train_res, y_train_res
    

def process_test(X_test, y_test):

        X_test=X_test[['emp_length_int','annual_inc',
                        'loan_amount','interest_rate','dti',
                        'total_rec_prncp','recoveries','home_ownership','income_category',
                            'term','application_type','purpose','interest_payments',
                            'grade','region']]    

    
        categorical=X_test[['home_ownership','income_category',
                            'term','application_type','purpose','interest_payments',
                            'grade','region']]
        

        #create label encoder categorical data on X_test
        le=LabelEncoder()
        X_test[categorical.columns]=X_test[categorical.columns].apply(lambda col: le.fit_transform(col))
    
        #perform random under sampling
        rus=RandomUnderSampler(random_state=42)
        X_test_res,y_test_res=rus.fit_resample(X_test,y_test)
    
        print("Data test processed")
    
        return X_test_res, y_test_res
   