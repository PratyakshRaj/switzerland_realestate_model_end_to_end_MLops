#modular code with exception handling 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import json
import mlflow
import dagshub
import mlflow.pyfunc
import cloudpickle



dagshub.init(repo_owner='PratyakshRaj', repo_name='switzerland_realestate_model_end_to_end_MLops', mlflow=True)
mlflow.set_experiment("Final_model")
mlflow.set_tracking_uri("https://dagshub.com/PratyakshRaj/switzerland_realestate_model_end_to_end_MLops.mlflow")


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}:{e}")

#test_data=pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data:pd.DataFrame)-> tuple[pd.DataFrame,pd.Series]:
    try:
      x=data.drop(['age of change in days','Ownership type','address','surface of ammenity','Construction year range','price'],axis=1)
      y=data["price"]
      return x,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}") 
#x_test= test_data.drop(['age of change in days','Ownership type','address','surface of ammenity','Construction year range','price'],axis=1)
#y_test=test_data["price"]

def load_model(filepath):
    try:
        with open(filepath,"rb") as file:
          model=pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")
    

#model=pickle.load(open("model.pkl","rb"))

def evaluation_model(model,x_test,y_test):
    try:
        y_pred=model.predict(x_test)


        r2_score_=r2_score(10**y_test,10**y_pred)
        root_mse=((mean_squared_error(10**y_test,10**y_pred))**0.5)

        metrics_dict={
            "r2_score":r2_score_,
            "root_mean_square_error":root_mse
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")

def save_metrics(metrics_dict,filepath):
    try:
        with open(filepath,"w") as file:
            json.dump(metrics_dict,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}") 

def main():
    try:
        test_data_path="./data/processed/test_processed.csv"
        model_path="models/model.pkl"
        metrics_path="reports/metrics.json"
        
        test_data=load_data(test_data_path)
        x_test,y_test=prepare_data(test_data)
        model=load_model(model_path)
        
        with mlflow.start_run() as run:
            metrics=evaluation_model(model,x_test,y_test)
            save_metrics(metrics,metrics_path) 
            
            mlflow.log_artifact(model_path) 
            mlflow.log_artifact(metrics_path) 
            
            mlflow.log_artifact(__file__)
            
            mlflow.log_metrics(metrics)
            mlflow.log_params({"min_data_in_leaf":  5,"iterations": 1200,"depth": 8,"colsample_bylevel": 1})
            
            sig=mlflow.models.infer_signature(x_test,model.predict(x_test))
            

            class CATWrapper(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    self.model = cloudpickle.load(open(context.artifacts["model"], "rb"))

                def predict(self, context, model_input):
                    return self.model.predict(model_input)

            mlflow.pyfunc.log_model(
                artifact_path="CatboostModel",
                python_model=CATWrapper(),
                artifacts={"model": "models/model.pkl"},
                signature=sig
            )
            
            run_info={'run_id':run.info.run_id,'model_name':"CatboostModel"}
            reports_path="reports/run_info.json"
            with open(reports_path,'w') as file:
                json.dump(run_info,file,indent=4)
                
    except Exception as e:
        raise Exception(f"An error occured:{e}")
    
if __name__=="__main__":
    main()               
            