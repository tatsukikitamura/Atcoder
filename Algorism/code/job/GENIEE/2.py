import requests

url = "http://localhost:3000/api/sns/"

N,M,L = map(input().split())

res = requests.get(url+"user?user_id={N}")
if res.status_code == 404:
    print("No user.")
else:
    data = res.json()
    print(data)