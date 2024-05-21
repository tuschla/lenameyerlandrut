import torch, cv2, os
import numpy as np


class BuildingSegmentation(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        config_json_path,
        image_count=0,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = np.array(sorted(os.listdir(images_dir)))
        self.mask_ids = np.array(sorted(os.listdir(masks_dir)))

        if image_count:
            random_idx = np.random.choice(
                len(self.ids), min(image_count, len(self.ids)), replace=False
            )
            self.ids = self.ids[random_idx]
            self.mask_ids = self.mask_ids[random_idx]

        self.image_files = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.mask_files = [
            os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids
        ]

        self.classes = {
            "rail": {"id": 1, "dim": 0, "color": [0, 120, 230]},
        }

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        # TODO: make slices / cutouts of the Tensors
        image = cv2.imread(self.image_files[index])
        mask = cv2.imread(self.mask_files[index], 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = [(mask == v["dim"]) for v in self.classes.values()]
        rails = np.bitwise_or.reduce(masks[-2:])
        mask = np.stack([(rails)], axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)
