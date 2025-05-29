import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pickle
import yaml

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}:{e}")

#train_data=pd.read_csv("./data/processed/train_processed.csv")

def prepare_data(data:pd.DataFrame)-> tuple[pd.DataFrame,pd.Series]:
    try:
      x=data.drop(['age of change in days','Ownership type','address','surface of ammenity','Construction year range','price'],axis=1)
      y=data["price"]
      return x,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}") 
#x_train= train_data.drop(['age of change in days','Ownership type','address','surface of ammenity','Construction year range','price'],axis=1)
#y_train=train_data["price"]


def load_params(params_path:str)->int:
    try:
        with open(params_path,"r") as file:
            params=yaml.safe_load(file)
        return [params["model_building"]["min_data_in_leaf"],params["model_building"]["iterations"],params["model_building"]["depth"],params["model_building"]["colsample_bylevel"]]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}:{e}")    
# depth=yaml.safe_load(open("params.yaml","r"))["model_building"]["depth"]
def train_model(x,y,para):
    try:
        cat_features = [i for i, col in enumerate(x.columns) if x[col].dtype == 'object']
        trained_model= CatBoostRegressor(min_data_in_leaf= para[0],iterations=para[1] ,depth= para[2],colsample_bylevel= para[3]).fit(x,y,cat_features=cat_features)
        return trained_model
    except Exception as e:
        raise Exception(f"Error training the model: {e}")

def save_model(model,filepath):
    try:
        with open(filepath,"wb") as file:
          pickle.dump(model,file)
    except Exception as e:
        raise Exception(f"Error saving model to {filepath}:{e}") 
       
#pickle.dump(trained_model,open("model.pkl","wb"))

def main():
    try:
        params_path="params.yaml"
        data_path = "./data/processed/train_processed.csv"
        model_name="models/model.pkl"
        
        para= load_params(params_path)
        train_data=load_data(data_path)
        x_train,y_train=prepare_data(train_data)
        
        model=train_model(x_train,y_train,para)
        save_model(model,model_name)
    except Exception as e:
        raise Exception(f"An error occured:{e}") 
    
if __name__=="__main__":
    main()   