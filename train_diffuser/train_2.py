from diffusers import UNet2DModel, DDPMPipeline, DDIMPipeline, DDIMScheduler
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from data_reader import dataloader
from evaluate_2 import evaluate
from evaluate_acce import evaluate_acce
from accelerate import Accelerator
from tqdm.auto import tqdm

import os
from accelerate import notebook_launcher
import glob
from data_reader import transform
from datasets import load_dataset
# import matplotlib.pyplot as plt
from trainConfig import TrainingConfig
import torch
from diffusers import UNet2DModel


# def is_parallel(model):
#     return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

# def de_parallel(model):
#     return model.module if is_parallel(model) else model



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.9 * averaged_model_parameter + 0.1 * model_parameter
    # ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

    ema_model = torch.optim.swa_utils.AveragedModel(model,
                                                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))


    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs_2"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()


                ema_model.update_parameters(model)

                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # model = ema_model.module
        # model = ema_model
        ema_model.to('cpu')
        if accelerator.is_main_process:

            pipeline_ddpm = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            ema_pipeline_ddpm = DDPMPipeline(unet=accelerator.unwrap_model(ema_model.module), scheduler=noise_scheduler)

            noise_scheduler_acce = DDIMScheduler(num_train_timesteps=1000)
            # pipeline_ddim = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler_acce)
            ema_pipeline_ddim = DDIMPipeline(unet=accelerator.unwrap_model(ema_model.module), scheduler=noise_scheduler_acce)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1 or epoch == 0:
            #     # evaluate(config, epoch, ema_pipeline_ddpm)
            #     # evaluate_acce(config, epoch, ema_pipeline_ddim)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1 or epoch == 0:
                pipeline_ddpm.save_pretrained(config.output_dir)
                ema_pipeline_ddpm.save_pretrained(config.output_dir_ema)





if __name__ == "__main__":
    config = TrainingConfig()
    dataset = load_dataset(config.dataset_name, split="train")
    print("Dataset loaded")
    # dataset = load_dataset(config.dataset_name)
    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    print(config.dataset_name)
    print(len(dataset))
    print(len(train_dataloader))

    model = UNet2DModel(
        sample_size=config.image_size,
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
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )





    # sample_image = dataset["images"].unsqueeze(0)
    sample_image = dataset[0]["images"].unsqueeze(0)
    # sample_image = dataset[0].unsqueeze(0)
    print(sample_image)
    print("Input shape:", sample_image.shape)
    print("Output shape:", model(sample_image, timestep=0).sample.shape)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([50])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    noise_pred = model(noisy_image, timesteps).sample
    loss = F.mse_loss(noise_pred, noise)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    notebook_launcher(train_loop, args, num_processes=6)

    sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))

