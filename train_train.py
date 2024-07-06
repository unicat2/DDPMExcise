import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
import data_reader
import sample
from ddpm import DDPM
from ddim import DDIM
# from network import UNet
# import Unet
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

batch_size = 32
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model_path = ''
save_dir = 'model_checkpoints_2'
log_dir = 'log_2'
n_timesteps = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
dataset_name = 'celeba'
xt_shape = data_reader.get_img_shape(dataset_name)


def train(ddpm, net):
    n_timesteps = ddpm.n_timesteps
    net = net.to(device)

    dataloader = data_reader.dataloader(dataset_name, batch_size)

    optimizer = torch.optim.Adam(net.parameters(), 1e-4)
    writer = SummaryWriter(log_dir)
    losses = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_time = time.time()
    for epoch in range(epochs):
        net.train()

        for i, (x, _) in enumerate(dataloader):

            x = x.to(device)
            timesteps = torch.randint(0, n_timesteps, (x.shape[0],)).to(device)
            noise = torch.randn_like(x).to(device)

            xt = noise_scheduler.add_noise(x, noise, timesteps)
            # xt = ddpm.p_forward(x, timesteps, noise)

            noise_pred = net(xt, timesteps, return_dict=False)[0]

            # if hasattr(noise_pred, 'sample'):
            #     noise_pred = noise_pred.sample
            # else:
            #     noise_pred = noise_pred
            #
            #
            # if not isinstance(noise_pred, torch.Tensor):
            #     noise_pred = torch.tensor(noise_pred).to(device)


            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

        loss_last_epoch = sum(losses[-len(dataloader):]) / len(dataloader)
        print(f"Epoch:{epoch + 1}, loss: {loss_last_epoch}")

        with open('loss.txt', 'a') as f:
            f.write(f"Epoch:{epoch + 1}, loss: {loss_last_epoch}\n")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(save_dir, f'unet_epoch_{epoch + 1}_{dataset_name}.pth')
            torch.save(net.state_dict(), model_save_path)
        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time cost: ", elapsed_time)
    writer.close()
    return losses


def generate(model, noise_scheduler):
    image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline_output = image_pipe()
    return pipeline_output.images[0]


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())

    net = UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # net = UNet(n_steps, img_shape, cfg['channels'], cfg['pe_dim'],
    #            cfg.get('with_attn', False), cfg.get('norm_type', 'ln'))

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ddpm = DDPM(net, n_timesteps, device)
    ddim = DDIM(net, n_timesteps, device)

    train(ddpm, net)






