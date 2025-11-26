import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()


os.environ['MLFLOW_TRACKING_URI']=os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME']=os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD']=os.getenv('MLFLOW_TRACKING_PASSWORD')

params=yaml.safe_load(open("params.yaml"))["train"]

def evalute(data_path,model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model=pickle.load(open(model_path,'rb'))

    prediction = model.predict(x)
    accuracy = accuracy_score(y,prediction)

    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__== "__main__":
    evalute(params["data"],params["model"])
    


