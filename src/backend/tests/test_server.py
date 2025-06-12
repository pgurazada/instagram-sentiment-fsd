import requests

BASE_URL = "http://127.0.0.1:8000/predict"

def test_negative_sentiment():
    response = requests.post(BASE_URL, json={"text": "This is a crappy phone"})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "sentiment_label" in data["data"]
    assert "sentiment_score" in data["data"]
    assert isinstance(data["data"]["sentiment_label"], str)
    assert isinstance(data["data"]["sentiment_score"], (int, float))

def test_positive_sentiment():
    response = requests.post(BASE_URL, json={"text": "I love this phone"})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "sentiment_label" in data["data"]
    assert "sentiment_score" in data["data"]
    assert isinstance(data["data"]["sentiment_label"], str)
    assert isinstance(data["data"]["sentiment_score"], (int, float))

def test_empty_text():
    response = requests.post(BASE_URL, json={"text": ""})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "sentiment_label" in data["data"]
    assert "sentiment_score" in data["data"]