# !pip install invisible_watermark transformers accelerate safetensors kornia
# !pip install git+https://github.com/huggingface/diffusers.git
# !pip install huggingface_hub



import torch
from torchvision.transforms import ToTensor, ToPILImage
from kornia.geometry.transform import get_affine_matrix2d, warp_affine


# Zooms out of a given image, and creates an outpainting mask for the external area.
def create_outpainting_image_and_mask(image, zoom):
    # 注意这里的zoom 是 扩大的比例, 如150% 
    # if zoom is not List
    if isinstance(zoom, list):
        new_h = int(image.size[1] * zoom[0])
        new_w = int(image.size[0] * zoom[1])
        # zoom = torch.tensor([1/zoom[0], 1/zoom[1]]).unsqueeze(0)
    elif isinstance(zoom, float):
        new_h = int(image.size[1] * zoom)
        new_w = int(image.size[0] * zoom)
        # zoom = torch.tensor([1/zoom, 1/zoom]).unsqueeze(0)
    else:
        assert False, "zoom should be a float or a list of two floats"
    image_tensor = ToTensor()(image).unsqueeze(0)
    _, c, h, w = image_tensor.shape

    center = torch.tensor((0., 0.)).unsqueeze(0)
    # move to the center
    translate = torch.tensor([new_w/2 - w/2, new_h/2 - h/2]).unsqueeze(0)
    angle = torch.tensor([0.0])
    zoom = torch.tensor([1., 1.]).unsqueeze(0)

    M = get_affine_matrix2d(
        center=center, translations=translate, angle=angle, scale=torch.tensor([1., 1.]).unsqueeze(0)
    )

    mask_image_tensor = warp_affine(
        image_tensor,
        M=M[:, :2],
        dsize=(new_h, new_w),
        padding_mode="fill",
        fill_value=-1*torch.ones(3),
    )
    mask = torch.where(mask_image_tensor < 0, 1.0, 0.0)

    transformed_image_tensor = warp_affine(
        image_tensor,
        M=M[:, :2],
        dsize=(new_h, new_w),
        padding_mode="border"
    )

    output_mask = ToPILImage()(mask[0])
    output_image = ToPILImage()(transformed_image_tensor[0])

    return output_image, output_mask

import PIL
from diffusers.utils import load_image

init_image = load_image("images/outpainting1.png")

# init_image = init_image.resize((512, 512))
conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, 1.5)
init_image.show()
conditioning_image.show()
outpaint_mask.show()
print(init_image.size)
print(conditioning_image.size)
print(outpaint_mask.size)


from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import Image

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")


def outpainting_pipe(init_image, zoom):

    # init_image = init_image.resize((512, 512))
    conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, zoom)

    prompt = ""
    generator = torch.Generator()
    seed = 12345

    #  `height` and `width` have to be divisible by 8, adjust closest the 1024

    # height = 1024
    # width = int(1024 * (conditioning_image.width / conditioning_image.height))
    # align the short side to 1024
    short_side = 1024
    if conditioning_image.width < conditioning_image.height:
        width = short_side
        height = int(short_side * (conditioning_image.height / conditioning_image.width)) // 8 * 8
    else:
        height = short_side
        width = int(short_side * (conditioning_image.width / conditioning_image.height)) // 8 * 8


    output = pipe(
        prompt,
        image=conditioning_image,
        mask_image=outpaint_mask,
        height=height,
        width=width,
        generator=generator.manual_seed(seed),
        strength=0.99,
    )
    return output.images[0], init_image, conditioning_image, outpaint_mask

def outpainting_pipe_times(init_image, zoom, times):
    image = init_image
    zoom_step = [zoom[0] ** (1 / times), zoom[1] ** (1 / times)]
    for i in range(times-1):
        image, _, _, _ = outpainting_pipe(image, zoom_step)
        image.show()
        # 去掉边缘30 个 pixels
        image = image.crop((30, 30, image.width-30, image.height-30))
        image.show()
    image, _, _, _ = outpainting_pipe(image, zoom_step)
    return image

init_image = load_image("images/outpainting1.png")
output_image, _, _, _ = outpainting_pipe(init_image, [4, 3])
output_image.save('out_image.png')
# import os
# image_dir = 'images/outpainting/'
# os.makedirs('images/outpainting_results', exist_ok=True)
# os.makedirs('images/outpainting_condition', exist_ok=True)
# os.makedirs('images/outpainting_mask', exist_ok=True)
# for image_name in os.listdir(image_dir):
#     image_path = os.path.join(image_dir, image_name)
#     init_image = load_image(image_path)
#     output_image, init_image, conditioning_image, outpaint_mask = outpainting_pipe(init_image, 1.5 )
#     output_image.save(f'images/outpainting_results/{image_name}_affine_1_5.png')
#     conditioning_image.save(f'images/outpainting_condition/{image_name}_affine_1_5_conditioning.png')
#     outpaint_mask.save(f'images/outpainting_mask/{image_name}_affine_1_5_mask.png')
#     output_image, init_image, conditioning_image, outpaint_mask = outpainting_pipe(init_image, [4, 3] )
#     output_image.save(f'images/outpainting_results/{image_name}_affine_4_3.png')
#     conditioning_image.save(f'images/outpainting_condition/{image_name}_affine_4_3_conditioning.png')
#     outpaint_mask.save(f'images/outpainting_mask/{image_name}_affine_4_3_mask.png')
#     # output_image = outpainting_pipe_times(init_image, [4, 3], 2)
#     # output_image.save(f'images/outpainting/{image_name}_affine_4_3_x2.png')
#     # output_image = outpainting_pipe_times(init_image, [4, 3], 3)
#     # output_image.save(f'images/outpainting/{image_name}_affine_4_3_x3.png')
#     # output_image = outpainting_pipe_times(init_image, [4, 3], 5)
#     # output_image.save(f'images/outpainting/{image_name}_affine_4_3_x5.png')
    