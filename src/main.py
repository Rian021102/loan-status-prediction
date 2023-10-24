from datapipeline import loaddata, new_eda,clean_train,clean_test
from binningIV import create_binning
import pandas as pd
import numpy as np
from feateng import process_train, process_test
from xtremegradientboost import xgb_model
def main():
    # Load the data
    path='/Users/rianrachmanto/pypro/data/loan_final313.csv'
    X_train, X_test, y_train, y_test = loaddata(path)
    print('Data loaded')

    # Explore the data
    eda_train=new_eda(X_train, y_train)
    X_train,y_train=clean_train(X_train,y_train)
    X_test,y_test=clean_test(X_test,y_test)
    print('Data cleaned')

    #create data_train
    X_traincop=X_train.copy()
    y_traincop=y_train.copy()
    y_traincop.columns=['loan_condition_cat']
    data_train=pd.concat([X_traincop,y_traincop],axis=1)

    #prepare for binning
    num_columns=data_train[['emp_length_int','annual_inc',
                            'loan_amount','interest_rate','dti',
                            'total_rec_prncp','recoveries']]
    
    cat_columns=data_train[['home_ownership','income_category',
                        'term','application_type','purpose','interest_payments',
                        'grade','region']]
    
    respon_var=data_train['loan_condition_cat']
    #create binning
    for col in num_columns:
        data_train_binned=create_binning(data=data_train, predictors=col, num_of_bins=10)
    print(data_train_binned.head(10).T)


    crosstab_num=[]

    for column in num_columns:
        crosstab=pd.crosstab(data_train_binned[column+'_bin'], respon_var, margins=True)
        crosstab_num.append(crosstab)
    
    crosstab_cat=[]
    for column in cat_columns:
        crosstab=pd.crosstab(data_train_binned[column], respon_var,margins=True)
        crosstab_cat.append(crosstab)
    
    crosstab_list=crosstab_num+crosstab_cat

    print(crosstab_list)
    
    WOE_list=[]
    IV_list=[]

    IV_table=pd.DataFrame({'Characteristic:':[],
                       'information_value:':[]})


    for crosstab in crosstab_list:
    
        crosstab['p_good']=crosstab[0]/crosstab[0]['All']

        crosstab['p_bad']=crosstab[1]/crosstab[1]['All']

        crosstab['WOE']=np.log(crosstab['p_good']/crosstab['p_bad'])

        crosstab['contribution']=crosstab['WOE']*(crosstab['p_good']-crosstab['p_bad'])

        IV=crosstab['contribution'] [:-1].sum()

        add_IV={'Characteristic:':crosstab.index.name,
            'information_value:':IV}
        WOE_list.append(crosstab)
        IV_list.append(add_IV)

    IV_table=IV_table.append(IV_list,ignore_index=True)
    
    print(WOE_list, IV_table)

    strength=[]

    for IV in IV_table['information_value:']:
        if IV < 0.02:
            strength.append('useless for prediction')
        elif IV >= 0.02 and IV < 0.1:
            strength.append('weak predictor')
        elif IV >= 0.1 and IV < 0.3:
            strength.append('medium predictor')
        elif IV >= 0.3 and IV < 0.5:
            strength.append('strong predictor')
        else:
            strength.append('suspicious or too good to be true')

    IV_table=IV_table.assign(Strength=strength)
    print(IV_table.sort_values(by='information_value:',ascending=False))
    
    X_train_res,y_train_res=process_train(X_train,y_train)
    X_test_res,y_test_res=process_test(X_test,y_test)

    model=xgb_model(X_train_res,y_train_res,X_test_res,y_test_res)    

    
    
if __name__ == '__main__':
    main()