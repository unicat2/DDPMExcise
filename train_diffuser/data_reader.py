from datasets import load_dataset
# import matplotlib.pyplot as plt
from trainConfig import TrainingConfig
from torchvision import transforms
import torch
from diffusers import UNet2DModel


config = TrainingConfig()

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




preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def dataloader():
    dataset = load_dataset(config.dataset_name, split="train")
    dataset.set_transform(transform)
    return torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


if __name__ == '__main__':

    dataset = load_dataset(config.dataset_name, split="train")

    print("Dataset loaded")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:4]["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.show()


    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    sample_image = dataset[0]["images"].unsqueeze(0)
    print("Input shape:", sample_image.shape)

    print("Output shape:", model(sample_image, timestep=0).sample.shape)





























































