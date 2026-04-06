import urllib.request
import base64
import json

API_KEY = "AIzaSyDG4_1xqAIamDTRLgLZ6UyObrxNch6tsxE"

def test_vision(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    body = json.dumps({
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }).encode()

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print(result)

# Change this path to one of your plate images
test_vision(r"C:\PROJECTS\vehicle-verification\dataset\IMG_2303.jpg")
