from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def download_model():
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


if __name__ == "__main__":
    download_model()
