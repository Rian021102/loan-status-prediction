from loaddata import load_data
from eda import eda
from featuring import featuring
from training import train 
from modelbase import modeling
def main():
    # Load the data
    path='/Users/rianrachmanto/pypro/data/loan_final313.csv'
    X_train, X_test, y_train, y_test = load_data(path)
    print('Data loaded')

    # Explore the data
    eda(X_train, y_train)
    

    # Preprocess the data
    X_train_rus, X_test_num, y_train_rus = featuring(X_train, X_test, y_train)

    # Train the model
    train(X_train_rus, y_train_rus, X_test_num, y_test)
    print('Models trained')

    
    
if __name__ == '__main__':
    main()