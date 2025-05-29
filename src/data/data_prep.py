import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath :str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception (f"Error loading data from {filepath}:{e}")

#train_data=pd.read_csv("./data/raw/train.csv")
#test_data=pd.read_csv("./data/raw/test.csv")

def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    try:
        df.rename(columns={"annexe":"amenities","no_postal":"postcode","Unnamed: 17":"ratio","nom_npa":"Area","adresse":"address","EPOQUE_CONSTRUCTION":"Construction year range","surface habitable":"living area","surface de l'annexe":"surface of ammenity","revised price with time":"price"},inplace=True)
        df.dropna(subset=["living area"],inplace=True)
        df.drop(df[df['price']==0].index,inplace=True)
        df.drop(df[df['living area']==0].index,inplace=True)
        df['amenities'].fillna('none',inplace=True)
        df['postcode']= df['postcode'].astype('str')
        df['Construction year range']=df['Construction year range'].str.split().str[-1]
        df['Construction year range'].fillna("1980",inplace=True)
        df['Construction year range']=df['Construction year range'].astype(int)
        df["proprerty average height"].fillna(df["proprerty average height"].mean(),inplace=True)
        df["price"]=np.log10(df["price"])
    except Exception as e:
        raise Exception(f"Error in preprocessing data:{e}")    
    
    return df
    
"""train_data.rename(columns={"annexe":"amenities","no_postal":"postcode","Unnamed: 17":"ratio","nom_npa":"Area","adresse":"address","EPOQUE_CONSTRUCTION":"Construction year range","surface habitable":"living area","surface de l'annexe":"surface of ammenity","revised price with time":"price"},inplace=True)
test_data.rename(columns={"annexe":"amenities","no_postal":"postcode","Unnamed: 17":"ratio","nom_npa":"Area","adresse":"address","EPOQUE_CONSTRUCTION":"Construction year range","surface habitable":"living area","surface de l'annexe":"surface of ammenity","revised price with time":"price"},inplace=True)

train_data.dropna(subset=["living area"],inplace=True)
train_data.drop(train_data[train_data['price']==0].index,inplace=True)
train_data.drop(train_data[train_data['living area']==0].index,inplace=True)

test_data.dropna(subset=["living area"],inplace=True)
test_data.drop(test_data[test_data['price']==0].index,inplace=True)
test_data.drop(test_data[test_data['living area']==0].index,inplace=True)

train_data['amenities'].fillna('none',inplace=True)
train_data['postcode']= train_data['postcode'].astype('str')
train_data['Construction year range']=train_data['Construction year range'].str.split().str[-1]


test_data['amenities'].fillna('none',inplace=True)
test_data['postcode']= test_data['postcode'].astype('str')
test_data['Construction year range']=test_data['Construction year range'].str.split().str[-1]

train_data['Construction year range'].fillna("1980",inplace=True)

test_data['Construction year range'].fillna("1980",inplace=True)

train_data['Construction year range']=train_data['Construction year range'].astype(int)
test_data['Construction year range']=test_data['Construction year range'].astype(int)

train_data["proprerty average height"].fillna(np.mean(train_data["proprerty average height"]),inplace=True)
test_data["proprerty average height"].fillna(np.mean(test_data["proprerty average height"]),inplace=True)

train_data["price"]=np.log10(train_data["price"])
test_data["price"]=np.log10(test_data["price"])
"""

#data_path=os.path.join("data","processed")
#os.makedirs(data_path)

def save_data(df,filepath):
    try:
        df.to_csv(filepath,index=False)    
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}:{e}")
    
#train_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
#test_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)

def main():
    try:
        raw_data_path="./data/raw/"
        processed_data_path = "./data/processed"
        
        train_data=load_data(os.path.join(raw_data_path,"train.csv"))
        test_data=load_data(os.path.join(raw_data_path,"test.csv"))
        
        train_processed_data=preprocess(train_data)
        test_processed_data=preprocess(test_data)
        
        os.makedirs(processed_data_path)
        
        save_data(train_processed_data,os.path.join(processed_data_path,"train_processed.csv"))
        save_data(test_processed_data,os.path.join(processed_data_path,"test_processed.csv"))
    except Exception as e:
        raise Exception(f"An error occured:{e}")    
if __name__ == "__main__":
    main() 