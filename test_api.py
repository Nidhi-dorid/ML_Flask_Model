import requests

url = "http://localhost:5000/detect"

files = {
    "image": open("test.jpg", "rb")
}
headers = {
    "X-Internal-Secret": "dev-secret-change-in-prod"
}

response = requests.post(url, files=files, headers=headers)

print("Status Code:", response.status_code)
print("Response:", response.json())