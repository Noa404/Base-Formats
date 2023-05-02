import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#read in function
def read_data(path:str) -> pd.Dataframe:
    '''
    This is a base function to read in data.
    '''
    df = pd.read_csv(path)
    return df

def split(df:pd.DataFrame, target:str, ratio:float):
    X = df[df.columns[~df.columns.isin([target])]]
    y = df[[target]]
    '''
    setting x to the columns 
    and y to are target colmns
    because the target will be in the columns
    '''
    train_X, train_y, test_X, test_y, = train_test_split(X,y, test_size = ratio)
    return train_X, train_y, test_X, test_y

def train_tree(train_X:pd.DataFrame, train_y:pd.DataFrame):

    '''
    Creating the model using a decision tree.
    train data for X 
    train data for y
    Max depth can be changed depending on the data.]
    A random_state can also be added here to 
    '''
    mod = DecisionTreeRegressor(max_depth=3)
    mod.fit(train_X,train_y)
    return mod 
def train_XGB_Classifier(train_X:pd.DataFrame, train_y:pd.DataFrame):
    '''
    Takes the train data and runs it through the XGB model
    '''
    mod = XGBClassifier
    mod.fit

def predict(features:pd.DataFrame) -> np.ndarray:

    '''
    Predicts the outputs of the given model
    
    Args: 
        test the data in the form of reain date
    
    Returns:
        Predictions per position
    '''
    prob = predict(features.values.astype(float))
    pred = np.squeeze(prob, axis = 1)

    return pred

def eval(expression:str):

    
    
    
df = read_data('/Users/noahz/Downloads/datasets/kc_house_data.csv')

