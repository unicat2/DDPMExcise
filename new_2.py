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

batch_size = 64
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = 'model_checkpoints_128_3'
log_dir = 'log_128_3'
load_checkpoint = False
n_timesteps = 1000
dataset_name = 'celeba_128'
xt_shape = data_reader.get_img_shape(dataset_name)


def train(ddpm, net, load_checkpoint=False):
    n_timesteps = ddpm.n_timesteps
    optimizer = torch.optim.Adam(net.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    dataloader = data_reader.dataloader(dataset_name, batch_size, num_workers=4)
    net = net.to(device)

    start_epoch = 0
    checkpoint_files = [f for f in os.listdir(save_dir) if
                        f.startswith('unet_epoch_') and f.endswith(f'_{dataset_name}.pth')]
    if load_checkpoint and checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[2]))
        checkpoint_path = os.path.join(save_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Loaded model from checkpoint and train.")
    else:
        print("Starting training.")

    writer = SummaryWriter(log_dir)
    losses = []
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        net.train()
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            noise = torch.randn_like(x).to(device)

            timesteps = torch.randint(0, n_timesteps, (x.shape[0],)).to(device)
            xt = noise_scheduler.add_noise(x, noise, timesteps)
            # xt = ddpm.p_forward(x, timesteps, noise)

            noise_pred = net(xt, timesteps, return_dict=False)[0]

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

        if (epoch + 1) % 1 == 0:
            model_save_path = os.path.join(save_dir, f'unet_epoch_{epoch + 1}_{dataset_name}.pth')
            # torch.save(net.state_dict(), model_save_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss
            }, model_save_path)

        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time cost: ", elapsed_time)
    writer.close()
    return losses



if __name__ == '__main__':
    torch.cuda.empty_cache()
    print("CUDA available:", torch.cuda.is_available())


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    )


    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ddpm = DDPM(net, n_timesteps, device)
    # ddim = DDIM(net, n_timesteps, device)

    train(ddpm, net)






