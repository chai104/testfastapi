import uvicorn
from fastapi import FastAPI
import json
import requests
from typing import List
from typing import Any
from dataclasses import dataclass
import ollama
from typing import Any, Callable, Dict, List, Literal
from PIL import Image
from typhoon_ocr.ocr_utils import render_pdf_to_base64png, get_anchor_text
from io import BytesIO
import base64

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

PROMPTS_SYS = {
    "default": lambda base_text: (f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
}


def get_prompt(prompt_name: str) -> Callable[[str], str]:
    """
    Get a prompt template function for the specified prompt type.

    This function returns a callable that generates a prompt template based on the provided prompt name.
    The returned function takes extracted text as input and returns a formatted prompt string
    that can be used with OCR/vision models.

    Available prompt types:
    - "default": Creates a prompt for extracting text with tables in markdown format.
    - "structure": Creates a prompt for extracting text with tables in HTML format and image analysis.

    Args:
        prompt_name (str): The identifier for the desired prompt template ("default" or "structure").

    Returns:
        Callable[[str], str]: A function that takes extracted text and returns a formatted prompt.

    Examples:
        >>> prompt_fn = get_prompt("default")
        >>> formatted_prompt = prompt_fn("Sample extracted text")
        >>> print(formatted_prompt[:50])  # Print first 50 chars
        Below is an image of a document page along with its
    """
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")

@app.get("/")
async def root():
    return {"message": "CDGS OCR version 1.0 API"}
    
#create new /v1/images:annotate
@app.post("/v1/images:annotate")
async def v1_images_annotate(request: Root):
    client = ollama.Client(
        host='http://89.221.67.138:47909',
    )
    if (request.requests[0].image.content.startswith("/9j/")):
        responsefiletype ="jpg"
        filename ="att-27041.pdf"
        page_num =1
        task_type= "default"
        image_base64 = render_pdf_to_base64png("att-27041.pdf", page_num, target_longest_image_dim=1800)
        #image_pil = Image.open(BytesIO(base64.b64decode(image_base64)))
        #image_base64 = request.requests[0].image.content
        anchor_text = get_anchor_text(filename, page_num, pdf_engine="pdfreport", target_length=8000)

        # Retrieve and fill in the prompt template with the anchor_text
        prompt_template_fn = get_prompt(task_type)
        PROMPT = prompt_template_fn(anchor_text)
        #prompt_fn = get_prompt("default")
        #formatted_prompt = prompt_fn("Sample extracted text")
        #messages1 = [{
        #        "role": "user",
        #        "content": PROMPT,
        #        "images": [""],
        #    }],

        #print (messages1)
        response = client.chat(model="scb10x/typhoon-ocr-7b", messages=[{
               "role": "user",
                "content": PROMPT,
                "images": [image_base64],
            }])
        response = response['message']['content']

    elif (request.requests[0].image.content.startswith("iVBORw0K")):
        responsefiletype ="png"
    elif (request.requests[0].image.content.startswith("SUk")):
        responsefiletype ="tif"
    elif (request.requests[0].image.content.startswith("JVBE")):
        responsefiletype ="pdf"
    else:
        responsefiletype = "unknown"
    #if (request.requests[0].features[0].engine == "llm-typhoon"):
    #    response = "Typhoon-OCR-7b"
    return {"response": response ,"type": responsefiletype ,"prompt": PROMPT  }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)