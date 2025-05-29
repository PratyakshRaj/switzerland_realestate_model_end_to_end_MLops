import mlflow.pyfunc
import pandas as pd
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/PratyakshRaj/switzerland_realestate_model_end_to_end_MLops.mlflow")

model_name="CatboostModel"

try:
    client = mlflow.tracking.MlflowClient()
    
    versions=client.get_latest_versions(model_name,stages=["Production"])
    
    if versions:
        latest_version=versions[0].version
        run_id=versions[0].run_id
        print(f"latest version in production: {latest_version}, Run ID: {run_id}")
        
        logged_model=f'runs:/{run_id}/{model_name}'
        print("logged_model:",logged_model)


        data=pd.DataFrame([{
            'type':"Habitation un logement", 'amenities':"Garage", 'postcode':1255, 'Area':"Veyrier", 'living area':388.0,
       'surface of garden for houses':1243.0, 'Building height':3.9,
       'proprerty average height':1.0, 'number of floors':2, 'perimeter':158.0,
       'age in years':25.0 
        }])

        # Below whole code is available on "make prediction in mlflow UI" 

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        prediction=10**(loaded_model.predict(data))
        print("prediction:",prediction)
    
    else:
        print("No model foung in production stage")
            
except Exception as e:
    print(f"Error fetching model: {e}")