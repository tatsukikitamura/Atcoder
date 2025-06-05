##curl https://challenge-server.tracks.run/hotel-reservation/reservations -H "X-ACCESS-TOKEN: 0111e7a5-de02-4703-b0cc-b01a8a65c511" -X POST -H "Content-Type: application/json" -d '{"checkin":"2022-12-01","checkout":"2022-12-02","plan_id":30,"number":2}'
##GET http://localhost:3000/api/sns/user?user_id=<user_id>
import requests

url = "http://localhost:3000/api/sns/"

N,M,L = map(input().split())

res = requests.get(url+"user?user_id={N}")
if res.status_code == 404:
    print("No user.")
else:
    data = res.json()
    print(data)