import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import make_column_transformer
import seaborn as sns
import openpyxl
from dateutil import relativedelta
from datetime import datetime
import mlflow
import dagshub
from mlflow.models import infer_signature

dagshub.init(repo_owner='PratyakshRaj', repo_name='switzerland_realestate_model_end_to_end_MLops', mlflow=True)
mlflow.set_experiment("Experiment_3")
mlflow.set_tracking_uri("https://dagshub.com/PratyakshRaj/switzerland_realestate_model_end_to_end_MLops.mlflow")


f5=pd.read_excel("D:\exp_mlflow\data\data.xlsx")
f5.rename(columns={"annexe":"amenities","no_postal":"postcode","Unnamed: 17":"ratio","nom_npa":"Area","adresse":"address","EPOQUE_CONSTRUCTION":"Construction year range","surface habitable":"living area","surface de l'annexe":"surface of ammenity","revised price with time":"price"},inplace=True)

f5.dropna(subset=["living area"],inplace=True)
f5.drop(f5[f5['price']==0].index,inplace=True)
f5.drop(f5[f5['living area']==0].index,inplace=True)
#f5["ratio"]=np.log10(f5["ratio"])
#f5.drop(f5[(f5["ratio"]<(np.mean(f5["ratio"])-2*(np.std(f5["ratio"])))) | (f5["ratio"]>(np.mean(f5["ratio"])+2*(np.std(f5["ratio"]))))].index,axis=0,inplace=True)
f5['amenities'].fillna('none',inplace=True)
f5['postcode']= f5['postcode'].astype('str')
#batch2.drop('prix/m2',axis=1,inplace=True)
l=f5['Construction year range'].str.split().str[-1]
f5['Construction year range']=l


#f5['Construction year range']="31.12."+f5['Construction year range'] 
f5['Construction year range'].fillna("1980",inplace=True)
#f5.dropna(inplace=True)
"""age=[]
for i in batch2.index:
    k= batch2.loc[i,'date'].split(".")
    k=[int(x) for x in k]
 
    l= (batch2.loc[i,'Construction year range'].split("."))
    l=[int(x) for x in l]
 
    
    age.append(days(l,k))"""
 
#f5["Construction year range"]=age
#m=batch2['date'].str.split('.').str[-1]
#batch2['date']=m
#batch2['date']=batch2['date'].astype(int) 
f5['Construction year range']=f5['Construction year range'].astype(int)
#f5["plan_RF"]=f5['plan_RF'].astype(str)
#f5.drop(["address","Ownership type"],axis=1,inplace=True)
#m=f5['date'].str.split('.').str[-1]
#f5['date']=m
#f5["date"].fillna("2020",inplace=True)
#f5['date']=f5['date'].astype(int) 
f5["proprerty average height"].fillna(np.mean(f5["proprerty average height"]),inplace=True)

print("qqqqq")

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print("ccccccc")

f5.drop(['age of change in days','Ownership type','address',"surface of ammenity",'Construction year range'],axis=1,inplace=True)





'''f5["proprerty average height"].fillna(np.mean(f5["proprerty average height"]),inplace=True)
f5["Building height"].fillna(np.mean(f5["Building height"]),inplace=True)
f5["number of floors"].fillna(np.mean(f5["number of floors"]),inplace=True)
f5["date"].fillna(np.mean(f5["date"]),inplace=True)
f5["ratio"].fillna(np.mean(f5["ratio"]),inplace=True)'''

x_1= f5.drop(['price'],axis=1)
y_1=f5["price"]
y_1=np.log10(y_1)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, test_size=0.25, random_state=7)

####
train_data = x_train_1.copy()
train_data["target"] = y_train_1

test_data = x_test_1.copy()
test_data["target"] = y_test_1

####

from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
import pickle
param_grid={
        "iterations": [1200],
        "depth": [3,5,8],
        "colsample_bylevel": [0.05,0.3,1],
        "min_data_in_leaf": [5,20,50]
}
CV_etr = RandomizedSearchCV(estimator=CatBoostRegressor(), param_distributions=param_grid, cv=3,n_jobs=-1)

cat_features = [i for i, col in enumerate(x_train_1.columns) if x_train_1[col].dtype == 'object']

with mlflow.start_run(run_name="Hyperparameter tuning") as parent_run:

    CV_etr.fit(x_train_1, y_train_1, **{"cat_features": cat_features})
    
    for i in range(len(CV_etr.cv_results_["params"])):
        with mlflow.start_run(run_name=f"combination{i+1}",nested=True)as child_run:
            mlflow.log_params(CV_etr.cv_results_["params"][i])
            mlflow.log_metric("test_scores",CV_etr.cv_results_["mean_test_score"][i])
    
    print("Best parameters: ",CV_etr.best_params_)


    print("lllll")

    trained_model= CV_etr.best_estimator_
    trained_model.fit(x_train_1,y_train_1,cat_features=cat_features)

    pickle.dump(trained_model,open("model.pkl","wb"))      

    cat_model=pickle.load(open("model.pkl","rb"))      
    y_cat_1=cat_model.predict(x_test_1)

    y_cat_1=10**(y_cat_1)
    
    mlflow.log_params(CV_etr.best_params_)
    mlflow.log_params({"cat_features":str(cat_features)})
    mlflow.log_metric("r2_score",r2_score(10**y_test_1,y_cat_1))
    mlflow.log_metric("rmse",(mean_squared_error(10**y_test_1,y_cat_1))**0.5)

    sig=infer_signature(x_test_1,CV_etr.best_estimator_.predict(x_test_1))
    
    import mlflow.pyfunc
    import cloudpickle

    class CATWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.model = cloudpickle.load(open(context.artifacts["model"], "rb"))

        def predict(self, context, model_input):
            return self.model.predict(model_input)

    mlflow.pyfunc.log_model(
        artifact_path="CstboostModel",
        python_model=CATWrapper(),
        artifacts={"model": "model.pkl"},
        signature=sig
    )
    
    mlflow.log_artifact(__file__)
    
    train_df=mlflow.data.from_pandas(train_data)
    test_df=mlflow.data.from_pandas(test_data)
       
    mlflow.log_input(train_df,"train")   
    mlflow.log_input(test_df,"test")   
        
    mlflow.set_tag("author","Pratyaksh")    
    mlflow.set_tag("model","Catboost")
    
         
    print("r2_score: ",r2_score(10**y_test_1,y_cat_1))
    print('rmse: ',(mean_squared_error(10**y_test_1,y_cat_1))**0.5)
    
    