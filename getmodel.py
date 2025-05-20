import os
import requests

os.makedirs('models', exist_ok=True)

files = {
    "decoder.pth": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth",
    "vgg_normalised.pth": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
}

for filename, url in files.items():
    filepath = os.path.join('models', filename)
    if os.path.exists(filepath):
        print(f"{filename} already exists, skipping download.")
        continue
    print(f"Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()  #raise error if download failed
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f"Saved {filename} to models/")
