import requests

try:
    r = requests.get("https://planetarycomputer.microsoft.com/api/stac/v1", timeout=10)
    print("OK:", r.status_code)
except Exception as e:
    print("ERROR:", e)