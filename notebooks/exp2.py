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

dagshub.init(repo_owner='PratyakshRaj', repo_name='switzerland_realestate_model_end_to_end_MLops', mlflow=True)
mlflow.set_experiment("Experiment_2")
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

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
f5.drop(['age of change in days','Ownership type','address',"surface of ammenity",'Construction year range'],axis=1,inplace=True)
types = f5.dtypes #type of each feature in data: int, float, object
num = types[(types == int) | (types == float)] #numerical values are either type int or float
cat = types[types == object] #categorical values are type object
dum_list=[]
categorical_values = list(cat.index)

for i in categorical_values:
    feature_set = set(f5[i])
    feature_list = list(feature_set)
    dum_list.append(feature_list)
    for j in feature_set:
         f5.loc[f5[i] == j, i] = str(feature_list.index(j))
        
'''f5["proprerty average height"].fillna(np.mean(f5["proprerty average height"]),inplace=True)
f5["Building height"].fillna(np.mean(f5["Building height"]),inplace=True)
f5["number of floors"].fillna(np.mean(f5["number of floors"]),inplace=True)
f5["date"].fillna(np.mean(f5["date"]),inplace=True)
f5["ratio"].fillna(np.mean(f5["ratio"]),inplace=True)'''

x_1= f5.drop(['price'],axis=1)
y_1=f5["price"]
y_1=np.log10(y_1)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1, y_1, test_size=0.25, random_state=7)


from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
'''param_grid={
        "iterations": [500,1000,1500,2000,2500],
        "learning_rate": [0.001,0.01],
        "depth": [2,5,10],
        "subsample": [0.05,0.3,1],
        "colsample_bylevel": [0.05,0.3,1],
        "min_data_in_leaf": [5,20,60,100],
}
CV_etr = RandomizedSearchCV(estimator=CatBoostRegressor(), param_distributions=param_grid, cv=5,n_jobs=-1)
CV_etr.fit(x_1, y_1)
print(CV_etr.best_params_)'''

with mlflow.start_run():

    #depth=6


    cat_features = [i for i, col in enumerate(x_train_1.columns) if x_train_1[col].dtype == 'object']
    trained_model= CatBoostRegressor().fit(x_train_1,y_train_1,cat_features=cat_features)
        

    pickle.dump(trained_model,open("model.pkl","wb"))      

    cat_model=pickle.load(open("model.pkl","rb"))      
    y_cat_1=cat_model.predict(x_test_1)

    y_cat_1=10**(y_cat_1)

    mlflow.log_metric("r2_score",r2_score(10**y_test_1,y_cat_1))
    mlflow.log_metric("rmse",(mean_squared_error(10**y_test_1,y_cat_1))**0.5)

    mlflow.log_param("depth","default")
    
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
        artifacts={"model": "model.pkl"}
    )
    mlflow.log_artifact("model.pkl","catboost")
    mlflow.log_artifact(__file__)
        
    print("r2_score: ",r2_score(10**y_test_1,y_cat_1))
    print('rmse: ',(mean_squared_error(10**y_test_1,y_cat_1))**0.5)

