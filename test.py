import torch
import os
import glob
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import click
import albumentations as A

classes = {
    "buildings": {"id": 1, "dim": 255, "color": [0, 120, 230]},
}


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(
    preprocessing_fn=smp.encoders.get_preprocessing_fn("resnet101", "imagenet")
):
    transform = []
    if preprocessing_fn:
        transform.append(A.Lambda(image=preprocessing_fn))
    transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
    return A.Compose(transform)


def vis_preprocessing():
    transform = [
        # A.LongestMaxSize(720, cv2.INTER_AREA),
        A.PadIfNeeded(1872),  # cv2.BORDER_CONSTANT),
    ]
    return A.Compose(transform)


def save_segmentation_predictions(
    image_folder, destination_folder, model_weights_path, threshold, bitmask
):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        activation="sigmoid",
        classes=len(classes),
    )
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_pp = get_preprocessing()
    vis_pp = vis_preprocessing()

    for image_path in glob.glob(image_folder + "*"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = vis_pp(image=image)["image"]
        out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = model_pp(image=image)["image"]
        x = torch.from_numpy(image).to(device).unsqueeze(0)

        with torch.no_grad():
            pred = model.predict(x)

        # pred = pred.squeeze().cpu().numpy().transpose(1, 2, 0)
        # print(pred.shape)
        pred = pred.squeeze(0).cpu().numpy()
        pred = pred.transpose(1, 2, 0)

        for label in classes.keys():
            ch = classes[label]["dim"]
            color = classes[label]["color"]
            m = pred[:, :, ch].squeeze()
            m_idx = m >= threshold
            out[m_idx] = out[m_idx] * 0.6 + np.array(color) * 0.4

            mask = np.zeros_like(out, dtype=np.uint8)
            mask[m_idx] = 255
            mask_filename = f"segmentation_mask_{label}_deeplab" + os.path.basename(image_path).replace("camera_image", "")
            mask_save_path = os.path.join(destination_folder, mask_filename)
            cv2.imwrite(mask_save_path, mask)
            print(f"Segmentation mask {mask_filename} saved for class {label}.")

        if not bitmask:
            filename = os.path.basename(image_path)
            prediction_save_path = os.path.join(destination_folder, filename)
            cv2.imwrite(prediction_save_path, out)
            print(f"Segmentation prediction saved for {filename}.")


@click.command
@click.option(
    "-i",
    "--image",
    "image_folder",
    type=click.Path(dir_okay=True),
    help="Image file path",
)
@click.option(
    "-o",
    "--destination_folder",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default="./runs/",
    help="Save result to file",
)
@click.option(
    "-w",
    "--weights",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=True),
    default="./weights/classes_deeplabv3plus_best.pth",
    help="Model weights",
)
@click.option(
    "-t",
    "--threshold",
    type=click.FloatRange(0.1, 1.0),
    default=0.1,
    help="Confidence threshold",
)
@click.option(
    "-m",
    "--bitmask",
    type=click.BOOL,
    help="If true, save only bitmask. Save both annotated image and mask if false",
    default=True,
)
def main(image_folder, destination_folder, weights, threshold, bitmask):
    save_segmentation_predictions(image_folder, destination_folder, weights, threshold, bitmask)

if __name__ == "__main__":
    main()
