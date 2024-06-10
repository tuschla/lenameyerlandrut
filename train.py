import torch
import click
import albumentations as A
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from dataset import BuildingSegmentation

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
ENCODER = "resnet101"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "sigmoid"
PATIENCE = 5
LR_INITIAL = 1e-3
LR_DECAY = 0.9
LR_ON_PLATEAU_DECAY = 0.1
LR_MINIMAL = 1e-5
MIN_SCORE_CHANGE = 1e-3
APPLY_LR_DECAY_EPOCH = 30


@click.command
@click.option(
    "-n",
    "--epochs",
    type=click.IntRange(1, 10000),
    default=2000,
    help="Number of epoch to train",
)
@click.option(
    "--use_cpu", is_flag=True, help="Run training on CPU (will be much slower)"
)
@click.option(
    "--use_unet", is_flag=True, help="Use Unet model"
)
@click.option(
    "-d",
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Location of the dataset",
)
@click.option(
    "-v",
    "--val_split",
    type=click.FloatRange(0.1, 1.0),
    default=0.3,
    help="Validation-to-train split ratio",
)
@click.option(
    "-s",
    "--save_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./weights/",
    help="Weights directory",
)
def main(epochs, use_cpu, use_unet, data_dir, val_split, save_dir):
    def get_training_augmentation():
        transform = [
            A.RandomCrop(height=16 * 23, width=16 * 40, always_apply=True),
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.VerticalFlip(p=0.5),  # Random vertical flip
            A.RandomRotate90(p=0.5),  # Random 90 degree rotation
            A.Transpose(p=0.5),  # Randomly transpose the image
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),  # Random shift, scale and rotate
            A.RandomBrightnessContrast(p=0.3),  # Random brightness and contrast
        ]
        return A.Compose(transform)

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype("float32")

    def get_preprocessing(preprocessing_fn=None):
        transform = []
        if preprocessing_fn:
            transform.append(A.Lambda(image=preprocessing_fn, always_apply=True))
        transform.append(A.Lambda(image=to_tensor, mask=to_tensor, always_apply=True))
        return A.Compose(transform)

    device = DEVICE_CUDA if not use_cpu else DEVICE_CPU
    # TODO: change this to match with the satellite image dataset
    images_dir = str(Path(data_dir).joinpath("jpgs", "rs19_val"))
    masks_dir = str(Path(data_dir).joinpath("uint8", "rs19_val"))
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset = BuildingSegmentation(
        images_dir,
        masks_dir,
        image_count=4200,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    model_unet = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=ACTIVATION,
        classes=1,
        # aux_params=dict(
        #     dropout=0.5,
        #     classes=len(dataset.classes),
        # ),
    )

    model_basic = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(128, 1, kernel_size=1, padding=0))

    n_train = int(len(dataset) * (1 - val_split))
    train_dataset, val_dataset = random_split(
        dataset, [n_train, len(dataset) - n_train]
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    loss = smp_utils.losses.DiceLoss()
    # loss = smp.losses.LovaszLoss(
    #    smp.losses.constants.MULTICLASS_MODE, per_image=False, ignore_index=None
    # )

    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
        smp_utils.metrics.Accuracy(threshold=0.5),
    ]

    if use_unet:
        model = model_unet
        optimizer = torch.optim.AdamW(
            [
                {"params": model.decoder.parameters(), "lr": LR_INITIAL},
            ]
        )

    else:
        model = model_basic
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": LR_INITIAL},
            ]
        )
        
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    val_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    writer = SummaryWriter()
    Path(save_dir).mkdir(exist_ok=True)
    save_name = str(Path(save_dir).joinpath("building_seg_unet.pth"))
    save_name_best = str(Path(save_dir).joinpath("building_seg_unet_best.pth"))

    min_score = 100
    min_score_epoch = epochs
    best_snapshot = None
    for epoch in range(0, epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch #{epoch+1} (learning rate - {lr:.2e})")

        train_logs = train_epoch.run(train_loader)
        valid_logs = val_epoch.run(val_loader)
        for metric_name, metric_value in train_logs.items():
            writer.add_scalar(f"Train/{metric_name}", metric_value, epoch)
        for metric_name, metric_value in valid_logs.items():
            writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch)

        score = valid_logs["dice_loss"]
        if score < min_score - MIN_SCORE_CHANGE:
            min_score = score
            min_score_epoch = epoch
            if epoch and epoch % APPLY_LR_DECAY_EPOCH == 0:
                lr *= LR_DECAY
            best_snapshot = model.state_dict().copy()
            torch.save(best_snapshot, save_name_best)

        elif min_score_epoch < epoch - PATIENCE:
            lr *= LR_ON_PLATEAU_DECAY
            min_score_epoch = epoch
        else:
            if epoch and epoch % APPLY_LR_DECAY_EPOCH == 0:
                lr *= LR_DECAY

        optimizer.param_groups[0]["lr"] = max(lr, LR_MINIMAL)


if __name__ == "__main__":
    main()
