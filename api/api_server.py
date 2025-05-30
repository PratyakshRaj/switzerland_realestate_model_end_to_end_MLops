from fastapi import FastAPI
import pickle
import pandas as pd
import mlflow
import mlflow.pyfunc
from data_model import realestate


app= FastAPI(
    title="Switzerland Realestate ML Model",
    description="Realestate price prediction"
)



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


        # Load model as a PyFuncModel.
        model = mlflow.pyfunc.load_model(logged_model)
        
    else:
        print("No model foung in production stage")
            
except Exception as e:
    print(f"Error fetching model: {e}")


@app.get("/")
def index():
    return " welcome to Switzerland's Realestate price prediction FastAPI "

@app.post("/predict")
def model_predict(realestate_obj:realestate):
    sample=pd.DataFrame([{
        'type':realestate_obj.type,
        'amenities':realestate_obj.amenities, 
        'postcode':realestate_obj.postcode, 
        'Area':realestate_obj.Area,
        'living area':realestate_obj.living_area,
        'surface of garden for houses':realestate_obj.surface_of_garden_for_houses,
        'Building height':realestate_obj.Building_height,
        'proprerty average height':realestate_obj.proprerty_average_height,
        'number of floors':realestate_obj.number_of_floors,
        'perimeter':realestate_obj.perimeter,
       'age in years':realestate_obj.age_in_years
    }])
    
    predicted_value=model.predict(sample)
    
    return "price is "+str(round(float(10**(predicted_value[0])),2))

    