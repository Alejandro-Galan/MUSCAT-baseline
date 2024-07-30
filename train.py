import gc

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from transformer.model import Transformer
from data.ar_dataset import ARDataModule
from utils.seed import seed_everything

seed_everything(42, deterministic=False, benchmark=False)

def train(
    ds_name,
    krn_encoding: str = "bekern",
    input_modality: str = "audio",  # "audio" or "image" or "both"
    use_distorted_images: bool = False,  # Only used if input_modality == "image" or "both"
    img_height: int = None,  # If None, the original image height is used (only used if input_modality == "image" or "both")
    attn_window: int = -1,
    epochs: int = 1000,
    patience: int = 20,
    batch_size: int = 16,
    checkpoint_path: str = "",
):
    gc.collect()
    torch.cuda.empty_cache()

    # TODO
    # Implement multimodal training
    if input_modality == "both":
        raise NotImplementedError("We can only train a unimodal model right now.")

    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tDataset: {ds_name}")
    print(f"\tKern encoding: {krn_encoding}")
    print(f"\tInput modality: {input_modality}")
    print(
        f"\tUse distorted images: {use_distorted_images} (used if input_modality in ['image', 'both'])"
    )
    print(f"\tImage height: {img_height} (used if input_modality in ['image', 'both'])")
    print(f"\tAttention window: {attn_window}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")

    # Data module
    datamodule = ARDataModule(
        ds_name=ds_name,
        krn_encoding=krn_encoding,
        input_modality=input_modality,
        use_distorted_images=use_distorted_images,
        img_height=img_height,
        batch_size=batch_size,
    )
    datamodule.setup(stage="fit")
    w2i, i2w = datamodule.get_w2i_and_i2w()
    max_h, max_w = datamodule.get_max_input_size()
    max_seq = datamodule.get_max_seq_len()



    # Model
    if not checkpoint_path:
        print("Not oading weights")
        model = Transformer(
            max_input_height=max_h,
            max_input_width=max_w,
            max_seq_len=max_seq,
            w2i=w2i,
            i2w=i2w,
            attn_window=attn_window,
            teacher_forcing_prob=0.2,
        )
    # Finetuning
    else:
        print("Loading weights", checkpoint_path)
        model = Transformer.load_from_checkpoint(checkpoint_path)



    model_name = input_modality
    model_name += (
        "_distorted" if input_modality == "image" and use_distorted_images else ""
    )
    model_name += (
        f"_height{img_height}"
        if input_modality == "image" and img_height is not None
        else ""
    )
    model_name += f"_{krn_encoding}"

    # Train, validate and test
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{ds_name}",
            filename=model_name,
            monitor="val_sym-er",
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor="val_sym-er",
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            check_on_train_epoch_end=False,
        ),
    ]
    # divergence_threshold=100.00,
    trainer = Trainer(
        logger=WandbLogger(
            project="OMR-A2S-Poly-Multimodal",
            group=model_name,
            name=f"Train-{ds_name}_Test-{ds_name}",
            log_model=False,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=5,
        deterministic=False,
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    
    model = Transformer.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
