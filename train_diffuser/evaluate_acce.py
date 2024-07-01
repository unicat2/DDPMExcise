from diffusers import DDPMPipeline, DDIMScheduler, DDIMPipeline
from diffusers.utils import make_image_grid
import os
import torch
from accelerate import Accelerator

def evaluate_acce(config, epoch, pipeline):

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed),
    ).images
    image_grid = make_image_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples_acce")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}_acce.png")



if __name__ == "__main__":
    noise_scheduler_acce = DDIMScheduler(num_train_timesteps=1000)
    pipeline_ddim = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler_acce)













