import uvicorn
from fastapi import FastAPI
import json
import requests
from typing import List
from typing import Any
from dataclasses import dataclass
import json

app = FastAPI()

@dataclass
class Feature:
    type: str
    engine: str
    output: str

    @staticmethod
    def from_dict(obj: Any) -> 'Feature':
        _type = str(obj.get("type"))
        _engine = str(obj.get("engine"))
        _output = str(obj.get("output"))
        return Feature(_type, _engine, _output)

@dataclass
class Image:
    content: str

    @staticmethod
    def from_dict(obj: Any) -> 'Image':
        _content = str(obj.get("content"))
        return Image(_content)

@dataclass
class Request:
    image: Image
    features: List[Feature]

    @staticmethod
    def from_dict(obj: Any) -> 'Request':
        _image = Image.from_dict(obj.get("image"))
        _features = [Feature.from_dict(y) for y in obj.get("features")]
        return Request(_image, _features)

@dataclass
class Root:
    requests: List[Request]

    @staticmethod
    def from_dict(obj: Any) -> 'Root':
        _requests = [Request.from_dict(y) for y in obj.get("requests")]
        return Root(_requests)

@app.get("/")
async def root():
    return {"message": "CDGS OCR version 1.0 API"}
    
#create new /v1/images:annotate
@app.post("/v1/images:annotate")
async def v1_images_annotate(request: Root):
    if (request.requests[0].features[0].engine == "llm-typhoon"):
        response = "Typhoon-OCR-7b"
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)