import os
from diffusers.utils import make_image_grid
import cv2
import einops
import numpy as np
import torchvision
from PIL import Image
import data_reader
from ddpm import DDPM
from ddim import DDIM
import torch
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, DDIMScheduler, DDIMPipeline

dataset_name = 'celeba_128'
# xt_shape = data_reader.get_img_shape(dataset_name)
xt_shape = (3, 128, 128)
n_timesteps = 1000
device = 'cuda'
model_path = './model_checkpoints_128_3/unet_epoch_65_celeba_128.pth'
n_sample = 64
save_dir = 'output_test_sample_V3'


def sample_my(ddpm_ddim, net, save_path="./tmp.jpg"):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *xt_shape)
        imgs = ddpm_ddim.q_backward(shape, net, device).detach().cpu()

        # imgs = (imgs + 1) / 2 * 255
        # imgs = imgs.clamp(0, 255)
        # imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample ** 0.5))
        # imgs = imgs.numpy().astype(np.uint8)
        # cv2.imwrite(save_path, imgs)
        grid_im = show_images(imgs)
        grid_im.save(save_path)


def sample_nopipeline_ddpm(net, noise_scheduler_ddpm, save_path="./tmp.jpg"):
    net = net.to(device)
    shape = (n_sample, *xt_shape)
    sample = torch.randn(shape).to(device)
    for i, t in enumerate(noise_scheduler_ddpm.timesteps):
        with torch.no_grad():
            residual = net(sample, t).sample
        sample = noise_scheduler_ddpm.step(residual, t, sample).prev_sample

    grid_im = show_images(sample)
    grid_im.save(save_path)



# def sample_nopipeline_ddim(net, noise_scheduler_ddim ,save_path="./tmp.jpg"):
#     net = net.to(device)
#     shape = (n_sample, *xt_shape)
#     sample = torch.randn(shape).to(device)
#     for i, t in enumerate(noise_scheduler_ddpm.timesteps):
#         with torch.no_grad():
#             residual = net(sample, t).sample
#         sample = noise_scheduler_ddim.step(residual, t, sample).prev_sample
#
#     grid_im = show_images(sample)
#     grid_im.save(save_path)


def sample_pipeline_ddpm(model, noise_scheduler, save_path="./tmp.jpg"):
    image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    images = image_pipe(
        batch_size=n_sample,
        generator=torch.Generator(device='cuda').manual_seed(0),
    ).images

    image_grid = make_image_grid(images, rows=8, cols=8)
    image_grid.save(save_path)

    # pipeline_output = image_pipe()
    # pipeline_output.images[0].save(save_path)

def sample_pipeline_ddim(model, noise_scheduler, save_path="./tmp.jpg"):
    image_pipe = DDIMPipeline(unet=model, scheduler=noise_scheduler)

    images = image_pipe(
        batch_size=n_sample,
        generator=torch.Generator(device='cuda').manual_seed(0),
    ).images

    image_grid = make_image_grid(images, rows=8, cols=8)
    image_grid.save(save_path)

    # pipeline_output = image_pipe()
    # pipeline_output.images[0].save(save_path)

def show_images(x):
    # img.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=128):
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net = UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
             "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(net), scheduler=noise_scheduler)


    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']

    # state_dict = torch.load(model_path)

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  # remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    net.load_state_dict(new_state_dict)

    # net.load_state_dict(torch.load(model_path))
    # net = UNet2DModel.from_pretrained("/model_checkpoints_1/unet_epoch_90_celeba.pth")

    save_path = os.path.join(save_dir, f'sample65_{dataset_name}.jpg')

    noise_scheduler_ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    noise_scheduler_ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    sample_pipeline_ddpm(net, noise_scheduler_ddpm, save_path.replace('.jpg', '_pipeline_ddpm.jpg'))
    sample_pipeline_ddim(net, noise_scheduler_ddim, save_path.replace('.jpg', '_pipeline_ddim.jpg'))

    sample_nopipeline_ddpm(net, noise_scheduler_ddpm, save_path.replace('.jpg', '_diff_ddpm.jpg'))

    # ddpm = DDPM(net, n_timesteps, device)
    # ddim = DDIM(net, n_timesteps, device)
    #
    # sample_my(ddpm, net, save_path.replace('.jpg', '_ddpm.jpg'))
    # sample_my(ddim, net, save_path.replace('.jpg', '_ddim.jpg'))
