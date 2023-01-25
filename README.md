
# 🍌 Banana Serverless instruct pix2pix

Mandatory inputs:
 - prompt: string
 - image: base64 image string (needs to be max 512x512)
  
Optional:
 - image_guidance: float default to 1.5
 - guidance_scale: float default to 7.5
 - negative_prompt: string default to None
 
 Returns:
 base64 image string
