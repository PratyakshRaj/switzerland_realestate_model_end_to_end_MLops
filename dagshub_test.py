import mlflow
import dagshub


mlflow.set_tracking_uri("https://dagshub.com/PratyakshRaj/switzerland_realestate_model_end_to_end_MLops.mlflow")


dagshub.init(repo_owner='PratyakshRaj', repo_name='switzerland_realestate_model_end_to_end_MLops', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
