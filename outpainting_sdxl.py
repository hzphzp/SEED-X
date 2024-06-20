from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import Image

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

def scale_and_paste(original_image):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 720
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 720
        new_width = round(new_height * aspect_ratio)

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    background = Image.new("RGB", (1024, 1024), "black")
    x = (1024 - new_width) // 2
    y = (1024 - new_height) // 2
    background.paste(resized_original, (x, y))
    pasted_image = background

    mask_image = Image.new("L", (1024, 1024), 255)
    mask_image.paste(Image.new("L", (new_width, new_height), 0), (x, y))

    return pasted_image, mask_image


original_image = load_image("images/face7.jpeg")
image, mask_image = scale_and_paste(original_image)
print("image size:", image.size)
print("mask_image size:", mask_image.size)
image = image.convert("RGB")

# image = load_image(img_url).resize((1024, 1024))
# mask_image = load_image(mask_url).resize((1024, 1024))
image.save("original_sdxl.png")
mask_image.save("mask_sdxl.png")

# prompt = "a tiger sitting on a park bench"
prompt = "President Obama giving a speech"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

image.save("outpainting_sdxl.png")



import random

import requests
import torch
from controlnet_aux import ZoeDetector
from PIL import Image, ImageOps

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
image_zoe = zoe(white_bg_image, detect_resolution=512, image_resolution=1024)
image_zoe.show()

controlnets = [
    ControlNetModel.from_pretrained(
        "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16
    ),
]
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnets, vae=vae
).to("cuda")