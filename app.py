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

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    image_base64 = model_inputs.get('image', None)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    if image_base64 == None:
        return {'message': "No image provided"}
    
    steps = model_inputs.get('steps', 30)
    image_guidance = model_inputs.get('image_guidance', 1.5)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    negative_prompt = model_inputs.get('negative_prompt', None)
    image_downloaded = PIL.Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    
    # Run the model
    images = model(prompt, image=image_downloaded, num_inference_steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, image_guidance_scale=image_guidance).images

    # Return the results as a dictionary
    buffered = BytesIO()
    images[0].save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
