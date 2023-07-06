import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

def featuring(X_train, X_test, y_train):
    # Take all variables from X_train and X_test
    #label encoding region both train and test
    le = LabelEncoder()
    X_train['region'] = le.fit_transform(X_train['region'])
    X_test['region'] = le.transform(X_test['region'])

    X_train_num = X_train.select_dtypes(include=['int64', 'float64']).copy()
    X_test_num = X_test.select_dtypes(include=['int64', 'float64']).copy()

    # Drop specific columns from X_train_num and X_test_num
    X_train_num.drop(['year', 'income_cat', 'installment', 'total_rec_prncp'], axis=1, inplace=True)
    X_test_num.drop(['year', 'income_cat', 'installment', 'total_rec_prncp'], axis=1, inplace=True)

    # Under sampling
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_num, y_train)

    # Print columns of X_train_num and X_test_num
    print('X_train_num columns:', X_train_num.columns)
    print('X_test_num columns:', X_test_num.columns)

    # Print shape of X_train_rus and y_train_rus
    print('X_train_rus shape:', X_train_rus.shape)
    print('y_train_rus shape:', y_train_rus.shape)

    return X_train_rus, X_test_num, y_train_rus
