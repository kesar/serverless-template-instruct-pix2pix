from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch

def download_model():
    model_id = "timbrooks/instruct-pix2pix"
    model = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    model.to("cuda")
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)

if __name__ == "__main__":
    download_model()
