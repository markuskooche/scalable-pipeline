from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_greeting():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Hello, FastAPI!'}


def test_predict_lower(lower_payload):
    response = client.post('/predict', json=lower_payload.model_dump())
    assert response.status_code == 200
    assert response.json() == {'salary': '<=50K'}


def test_predict_higher(higher_payload):
    response = client.post('/predict', json=higher_payload.model_dump())
    assert response.status_code == 200
    assert response.json() == {'salary': '>50K'}
