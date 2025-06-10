import requests

local_url = "http://127.0.0.1:8000/predict"

# local testing
response = requests.post(
    local_url,
    json={"text": "This is a crappy phone"}
)

print(response.json())