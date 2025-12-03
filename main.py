import pickle
import pandas as pd
from fastapi import FastAPI

from ml.data import process_data
from ml.model import inference
from schema import PredictPayload
from train_model import cat_features


with open('model/model.pkl', 'rb') as model_file:
    config = pickle.load(model_file, encoding='utf-8')

app = FastAPI()

@app.get('/')
async def greeting():
    return { 'message': 'Hello, FastAPI!' }

@app.post('/predict')
async def predict(payload: PredictPayload):
    df = pd.DataFrame([payload.model_dump()])
    df.rename(lambda key: key.replace('_', '-'), axis='columns', inplace=True)

    x, _, _, _ = process_data(df, cat_features, None, False, config['encoder'], config ['lb'])
    preds = inference(config['model'], x)

    return { 'salary': '>50K' if preds[0] else '<=50K' }
