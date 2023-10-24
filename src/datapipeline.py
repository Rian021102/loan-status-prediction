import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def loaddata(pathfile):
    df=pd.read_csv(pathfile)
    print(df.head())
    print(df.columns)
    #print unique id
    print(df['id'].nunique())
    #reset index
    df=df.reset_index(drop=True)

    #convert issue_d to datetime
    df['issue_d']=pd.to_datetime(df['issue_d'])

    #convert final_d to datetime
    df['final_d'] = pd.to_datetime(df['final_d'], format='%m%d%Y')


    print('Inspect the imbalance: ',df['loan_condition'].value_counts(normalize=True))    
   
    
    #set x and y
    X=df.drop(['loan_condition_cat'],axis=1)
    y=df['loan_condition_cat']
    #split data
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)

    return X_train,X_test,y_train,y_test


def new_eda(X_train,y_train):
    y_train.columns=['loan_condition_cat']
    eda_train=pd.concat([X_train,y_train],axis=1)
    numeric_cols=eda_train[['emp_length_int','annual_inc',
                            'loan_amount','interest_rate','dti','total_pymnt',
                            'total_rec_prncp','recoveries','installment']]

    #print missing values
    print(eda_train.isnull().sum())

    def describe_numeric(eda_train):
        for col in numeric_cols:
            print('Describe for column: ',col)
            print(eda_train[col].describe())
            plt.figure(figsize=(10,8))
            sns.histplot(data=eda_train,x=col)
            plt.show()
            plt.figure(figsize=(10,8))
            sns.boxplot(data=eda_train,x='loan_condition_cat', y=col)
            plt.show()
            plt.figure(figsize=(10,8))
            sns.kdeplot(data=eda_train,x=col, hue='loan_condition_cat')
            plt.show()
            print('Describe for column devided by loan condition:')
            print(eda_train.groupby('loan_condition_cat')[col].describe())
            print('-----------------------------------------------------------')
    describe_numeric(eda_train)

    plt.figure(figsize=(20,20))
    sns.heatmap(X_train.corr(),annot=True)
    plt.show()

    return eda_train

def clean_train(X_train, y_train):

    # Drop columns that cause multicollinearity
    X_train = X_train.drop(['year','interest_payment_cat', 'income_cat',
                          'installment', 'grade_cat', 'total_pymnt'], axis=1)
    
    num=X_train[['dti','annual_inc']]
    #remove outliers using iqr
    Q1=num.quantile(0.25)
    Q3=num.quantile(0.75)
    IQR=Q3-Q1
    print(IQR)
    X_train=X_train[~((num<(Q1-1.5*IQR))|(num>(Q3+1.5*IQR))).any(axis=1)]
    y_train=y_train.loc[X_train.index]

    print(X_train.shape, y_train.shape)
    return X_train, y_train

def clean_test(X_test, y_test):

    # Drop columns that cause multicollinearity
    X_test = X_test.drop(['year', 'interest_payment_cat', 'income_cat', 'installment', 'grade_cat', 'total_pymnt'], axis=1)
    
    return X_test, y_test