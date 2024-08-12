import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    x_train=pd.read_csv('/Users/nirbhaysedha/Desktop/C/data/interim/x_train.csv')
    x_test=pd.read_csv('/Users/nirbhaysedha/Desktop/C/data/interim/x_test.csv')
    y_train=pd.read_csv('/Users/nirbhaysedha/Desktop/C/data/interim/y_train.csv')
    y_test=pd.read_csv('/Users/nirbhaysedha/Desktop/C/data/interim/y_test.csv')
    return x_train,x_test,y_train,y_test

def train_model(x_train,x_test,y_train,y_test):
    model=RandomForestClassifier()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(accuracy_score(y_test,y_pred))
    joblib.dump(model,'model.pkl')


def main():
    x_train,x_test,y_train,y_test=load_data()
    train_model(x_train,x_test,y_train,y_test)

main()
