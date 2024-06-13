from diffusers import DiffusionPipeline
import torch
# hf_token = "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"

pipeline = DiffusionPipeline.from_pretrained("/home/t-zhiphuang/stable-diffusion-3-medium-diffusers/")

print('init pipeline done')

pipeline = pipeline.to('cuda')
prompt = "I need to generate an image, which is a bird's-eye view of the Guangzhou Tower, and it should be as high-definition as possible."
image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.0).images[0]
image.show()

pipeline = pipeline.to('cuda')
prompt = "Lu Xun is ridding a horse in the sunset."
image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.0).images[0]
image.show()