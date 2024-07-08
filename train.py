import torch
import matplotlib.pyplot as plt
import click
import albumentations as A
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from pathlib import Path
import itertools
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import BuildingSegmentation
import ray
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler


DEVICE_CUDA = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_CPU = "cpu"
ENCODER = "resnet101"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "sigmoid"
PATIENCE = 5
LR_DECAY = 0.9
LR_ON_PLATEAU_DECAY = 0.1
LR_MINIMAL = 1e-5
MIN_SCORE_CHANGE = 1e-3
APPLY_LR_DECAY_EPOCH = 30

AUGMENTATIONS = [
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(p=0.5),
    A.RandomScale(p=0.5),
    A.Affine(shear=20),
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3),
]


def generate_augmentation_combinations(augmentations):
    all_combinations = []
    for i in range(1, len(augmentations) + 1):
        combinations = itertools.combinations(augmentations, i)
        for combo in combinations:
            all_combinations.append(A.Compose(list(combo)))
    return all_combinations


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    transform = []
    if preprocessing_fn:
        transform.append(A.Lambda(image=preprocessing_fn, always_apply=True))
    transform.append(A.Lambda(image=to_tensor, mask=to_tensor, always_apply=True))
    return A.Compose(transform)


def train_model(config, augmentation=None):
    device = DEVICE_CUDA
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = BuildingSegmentation(
        Path(config["train_data_dir"]).joinpath("imgs"),
        Path(config["train_data_dir"]).joinpath("masks"),
        augmentation=augmentation,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    val_dataset = BuildingSegmentation(
        Path(config["val_data_dir"]).joinpath("imgs"),
        Path(config["val_data_dir"]).joinpath("masks"),
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataset = BuildingSegmentation(
        Path(config["test_data_dir"]).joinpath("imgs"),
        Path(config["test_data_dir"]).joinpath("masks"),
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    if config["use_unet"]:
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            activation=ACTIVATION,
            classes=1,
        )
    else:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    loss = smp_utils.losses.DiceLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
        smp_utils.metrics.Accuracy(threshold=0.5),
        smp_utils.metrics.Precision(threshold=0.5),
        smp_utils.metrics.Recall(threshold=0.5),
        smp_utils.metrics.Fscore(threshold=0.5),
    ]

    optimizer_class = getattr(torch.optim, config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["lr_initial"])

    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=False,
    )
    val_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=False,
    )

    writer = SummaryWriter()
    model_name = "unet" if config["use_unet"] else "basic"
    save_dir = f"./weights_lr_{config['lr_initial']}_bs_{config['batch_size']}_optimizer_{config['optimizer']}_{model_name}"
    Path(save_dir).mkdir(exist_ok=True)
    save_name_best = str(Path(save_dir).joinpath("best.pth"))

    min_score = 100
    min_score_epoch = 0
    best_snapshot = None
    for epoch in range(0, config["epochs"]):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch #{epoch+1} (learning rate - {lr:.2e})")

        train_logs = train_epoch.run(train_loader)
        valid_logs = val_epoch.run(val_loader)
        for metric_name, metric_value in train_logs.items():
            writer.add_scalar(f"Train/{metric_name}", metric_value, epoch)
        for metric_name, metric_value in valid_logs.items():
            writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch)

        score = valid_logs["iou_score"]
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

        # train.report({"dice_loss": score})

    # Load best model and evaluate
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(save_name_best, map_location=device))
    model.to(device)
    model.eval()

    logs = val_epoch.run(val_loader)
    print("Evaluation results:")
    train.report(logs)


def run_training_phase(config):
    scheduler = ASHAScheduler(
        metric="iou_score",
        mode="max",
        max_t=config["max_t"],
        grace_period=10,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_model,
        config=config,
        num_samples=config["num_samples"],
        scheduler=scheduler,
        resources_per_trial={"cpu": 4, "gpu": 0.5},
    )

    print("Initial phase best config: ", analysis.best_config)
    return analysis.best_config, analysis.results_df


def run_augmentation_phase(config, best_config):
    config.update(best_config)
    del config["num_samples"]

    augmentation_combinations = generate_augmentation_combinations(AUGMENTATIONS)

    for i, subset in enumerate(augmentation_combinations):
        aug_config = config.copy()
        aug_config["augmentation"] = subset
        scheduler = ASHAScheduler(
            metric="iou_score",
            mode="max",
            max_t=config["max_t"],
            grace_period=10,
            reduction_factor=2,
        )

        analysis = tune.run(
            train_model,
            config=aug_config,
            num_samples=1,
            scheduler=scheduler,
            resources_per_trial={"cpu": 4, "gpu": 0.5},
        )

        print(f"Augmentation phase subset {i} best config: ", analysis.best_config)
        return analysis.best_config, analysis.results_df


def plot_results(
    results_df,
    name=None,
    title=None,
    filter=["config/lr_initial", "config/batch_size", "config/optimizer", "dice_loss"],
):
    df_metrics = results_df[filter]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_metrics.values,
        colLabels=df_metrics.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df_metrics.columns))))

    if title:
        plt.title(title, fontsize=15, pad=20)

    plt.savefig(name)


@click.command()
@click.option(
    "--num_samples",
    type=click.IntRange(1, 5),
    default=1,
    help="Number of samples for hyperparameter search",
)
@click.option(
    "--max_epochs",
    type=click.IntRange(1, 10000),
    default=100,
    help="Maximum number of epochs for training",
)
@click.option(
    "--train_data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./segmentation_dataset_train",
    help="Location of the training dataset",
)
@click.option(
    "--val_data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./segmentation_dataset_val",
    help="Location of the validation dataset",
)
@click.option(
    "--test_data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./segmentation_dataset_test",
    help="Location of the testing dataset",
)
def main(num_samples, max_epochs, train_data_dir, val_data_dir, test_data_dir):
    train.RunConfig("./results")
    ray.init(num_cpus=8, num_gpus=1)

    no_tuning_config = {
        "max_t": 20,
        "num_samples": num_samples,
        "epochs": max_epochs,
        "use_unet": tune.grid_search([True, False]),
        "train_data_dir": Path(train_data_dir).resolve(),
        "val_data_dir": Path(val_data_dir).resolve(),
        "test_data_dir": Path(test_data_dir).resolve(),
        "batch_size": 32,
        "lr_initial": 1e-3,
        "optimizer": "AdamW",
    }

    best_config_no_tuning, results_df = run_training_phase(no_tuning_config)
    plot_results(
        results_df,
        name="no_tuning",
        title="Results without hyperparameter tuning lr: 1e-4, bs: 32, optim: AdamW.",
    )

    config = {
        "max_t": 20,
        "num_samples": num_samples,
        "epochs": max_epochs,
        "use_unet": tune.grid_search([True, False]),
        "train_data_dir": Path(train_data_dir).resolve(),
        "val_data_dir": Path(val_data_dir).resolve(),
        "test_data_dir": Path(test_data_dir).resolve(),
        "batch_size": tune.grid_search([16, 32, 64]),
        "lr_initial": tune.grid_search([1e-3, 1e-4, 1e-5]),
        "optimizer": tune.grid_search(["Adam", "SGD", "AdamW"]),
    }

    best_config, results = run_training_phase(config)
    plot_results(results, name="tuning", title="Results with hyperparameter tuning")

    best_aug_config, aug_results = run_augmentation_phase(config, best_config)
    plot_results(
        aug_results,
        name="augmentation",
        title="Results from trying augmentation combinations.",
        filter=[
            "config/lr_initial",
            "config/batch_size",
            "config/optimizer",
            'config/"augmentation"',
            "dice_loss",
        ],
    )


if __name__ == "__main__":
    main()
