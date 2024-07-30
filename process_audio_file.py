# Simplified version of the model. Receives .wav audio as input and returns adequate .kern format

import gc
import os

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from transformer.model import Transformer
from data.ar_dataset import ARDataModule
from utils.seed import seed_everything

seed_everything(42, benchmark=False)


def test(
    input_audio_folder,
    output_path_folder,
    krn_encoding: str = "bekern",
    checkpoint_path: str = "",
):
    gc.collect()
    torch.cuda.empty_cache()


    # Check if checkpoint path is empty or does not exist
    if checkpoint_path == "":
        raise ValueError("Checkpoint path not provided")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Get source dataset name
    _, src_ds_name, model_name = checkpoint_path.split("/")

    # Experiment info
    print("INFERENCE ON IMAGE")
    print(f"\tSource dataset: {src_ds_name}")
    print(f"\tTest dataset: {input_audio_folder}")
    print(f"\tKern encoding: {krn_encoding}")
    print(f"\tCheckpoint path: {checkpoint_path}")


    input_modality = "audio"  # "audio" or "image" or "both"
    use_distorted_images = False  # Only used if input_modality == "image" or "both"
    img_height = None

    # Data module
    datamodule = ARDataModule(
        ds_name=input_audio_folder,
        krn_encoding=krn_encoding,
        input_modality=input_modality,
        use_distorted_images=use_distorted_images,
        img_height=img_height,
        inference=True,
    )
    datamodule.setup(stage="predict")
    ytest_i2w = datamodule.test_ds.i2w

    # Model
    model = Transformer.load_from_checkpoint(checkpoint_path, ytest_i2w=ytest_i2w)

    # Test
    trainer = Trainer(
        precision="16-mixed",  # Mixed precision training
    )
    model.eval()
    breakpoint()
    output = trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(test)
