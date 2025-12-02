from fastapi import FastAPI

from schema import InputData

app = FastAPI()

@app.get('/')
async def greeting():
    return { 'message': 'Hello, FastAPI!' }

@app.post('/predict')
async def predict(data: InputData):
    return { 'message': str(data.foo) }
