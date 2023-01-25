import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

import base64
from io import BytesIO
import os

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    model.to("cuda")
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', 'blue skin')
    image_url = model_inputs.get('url', 'https://avatars.githubusercontent.com/u/1288106?s=400&u=7a91d67df3f308c76ceda288237b150d1bf859d2&v=4')
    image_downloaded = download_image(image_url)
    
    # Run the model
    images = model(prompt, image=image_downloaded, num_inference_steps=10, image_guidance_scale=1).images

    # Return the results as a dictionary
    buffered = BytesIO()
    images[0].save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
