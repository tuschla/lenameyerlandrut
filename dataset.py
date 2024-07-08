import torch, cv2, os
import numpy as np


class BuildingSegmentation(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        image_count=0,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = np.array(sorted(os.listdir(images_dir)))
        self.mask_ids = np.array(sorted(os.listdir(masks_dir)))

        if image_count:
            self.rng = np.random.default_rng()
            random_idx = self.rng.choice(len(self.ids), min(image_count, len(self.ids)), replace=False)
            self.ids = self.ids[random_idx]
            self.mask_ids = self.mask_ids[random_idx]

        self.image_files = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.mask_files = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        self.classes = {
            "building": {"id": 1, "dim": 255, "color": [0, 120, 230]},
        }

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = self.mask_files[idx]

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_file, 0)

        masks = [(mask == v["dim"]) for v in self.classes.values()]
        mask_combined = np.stack(masks, axis=-1)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask_combined)
            image, mask_combined = augmented["image"], augmented["mask"]

        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask_combined)
            image, mask_combined = preprocessed["image"], preprocessed["mask"]

        return image, mask_combined