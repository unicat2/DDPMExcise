from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 32
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 5
    mixed_precision = "fp16"  # `no`  float32, `fp16`  automatic mixed precision
    output_dir = "celeba-128-ema"
    output_dir_ema = "celeba-128-ema"
    seed = 0
    dataset_name = "lansinuote/gen.1.celeba"


