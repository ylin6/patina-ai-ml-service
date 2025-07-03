from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import clip
import numpy as np
from io import BytesIO
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(tensor).cpu().numpy()[0]

    return {"embedding": embedding.tolist()}

@app.post("/embed-url")
def embed_image_from_url(payload: dict = Body(...)):
    url = payload.get("url")
    if not url:
        return {"error": "Missing 'url' in payload"}

    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(tensor).cpu().numpy()[0]

        return {"embedding": embedding.tolist()}
    except Exception as e:
        return {"error": str(e)}
