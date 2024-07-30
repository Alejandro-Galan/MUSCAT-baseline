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
    ds_name,
    krn_encoding: str = "bekern",
    input_modality: str = "audio",  # "audio" or "image" or "both"
    use_distorted_images: bool = False,  # Only used if input_modality == "image" or "both"
    img_height: int = None,  # If None, the original image height is used (only used if input_modality == "image" or "both")
    checkpoint_path: str = "",
):
    gc.collect()
    torch.cuda.empty_cache()

    # TODO
    # Implement multimodal testing
    if input_modality == "both":
        raise NotImplementedError(
            "We can only perform unimodal model testing right now."
        )

    # Check if checkpoint path is empty or does not exist
    if checkpoint_path == "":
        raise ValueError("Checkpoint path not provided")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Get source dataset name
    _, src_ds_name, model_name = checkpoint_path.split("/")

    # Experiment info
    print("TEST EXPERIMENT")
    print(f"\tSource dataset: {src_ds_name}")
    print(f"\tTest dataset: {ds_name}")
    print(f"\tKern encoding: {krn_encoding}")
    print(f"\tInput modality: {input_modality}")
    print(
        f"\tUse distorted images: {use_distorted_images} (used if input_modality in ['image', 'both'])"
    )
    print(f"\tImage height: {img_height} (used if input_modality in ['image', 'both'])")
    print(f"\tCheckpoint path: {checkpoint_path}")

    # Data module
    datamodule = ARDataModule(
        ds_name=ds_name,
        krn_encoding=krn_encoding,
        input_modality=input_modality,
        use_distorted_images=use_distorted_images,
        img_height=img_height,
    )
    datamodule.setup(stage="test")
    ytest_i2w = datamodule.test_ds.i2w

    # Model
    model = Transformer.load_from_checkpoint(checkpoint_path, ytest_i2w=ytest_i2w)

    # Test
    trainer = Trainer(
        logger=WandbLogger(
            project="OMR-A2S-Poly-Multimodal",
            group=model_name.split(".ckpt")[0],
            name=f"Train-{src_ds_name}_Test-{ds_name}",
            log_model=False,
        ),
        precision="16-mixed",  # Mixed precision training
    )
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(test)
