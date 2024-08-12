import pathlib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(path,target,test):
    df=pd.read_csv(path)
    df.drop(columns=['Unnamed: 0','Aspect_Ratio'],inplace=True)
    df.rename(columns={'Length (major axis)':'length',
                       'Convex hull(convex area)':'Convex',
                       'Thickness (depth)':'Thickness',
                       'Width (minor axis)':'width'},
                       inplace=True)
    x=df.drop(columns=target)
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test,random_state=23)
    y_train=pd.DataFrame(y_train)
    y_test=pd.DataFrame(y_test)
    y_train.to_csv('./data/interim/y_train.csv',index=False)
    y_test.to_csv('./data/interim/y_test.csv',index=False)
    return x_train,x_test

def preprocess(x_train,x_test):
    imputer=SimpleImputer(strategy='mean')
    x_traini=imputer.fit_transform(x_train)
    x_testi=imputer.transform(x_test)
    scaler=MinMaxScaler()
    x_trains=scaler.fit_transform(x_traini)
    x_tests=scaler.transform(x_testi)
    x_train = pd.DataFrame(x_trains, columns=x_train.columns)
    x_test = pd.DataFrame(x_tests, columns=x_test.columns)
    x_train.to_csv('./data/interim/x_train.csv',index=False)
    x_test.to_csv('./data/interim/x_test.csv',index=False)

def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent
    data=home_dir.as_posix()+'/data/external/Almond.csv'
    target='Type'
    test=0.2
    x_train,x_test=load_data(data,target,test)
    x_traint,x_testt=preprocess(x_train,x_test)
    tt=pd.DataFrame(x_traint)
    print(tt.head())


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(f"Exception caught{e}")
