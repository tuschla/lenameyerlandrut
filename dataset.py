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
        patch_size=256,
        stride=256,
    ):
        self.ids = np.array(sorted(os.listdir(images_dir)))
        self.mask_ids = np.array(sorted(os.listdir(masks_dir)))

        if image_count:
            self.rng = np.random.default_rng()
            random_idx = self.rng.choice(len(self.ids), min(image_count, len(self.ids)), replace=False)
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

        self.patch_size = patch_size
        self.stride = stride

        # Precompute the list of patch indices for each image
        self.patch_indices = []
        for image_file in self.image_files:
            img_height, img_width = self._get_image_shape(image_file)
            num_patches_y = (img_height - self.patch_size) // self.stride + 1
            num_patches_x = (img_width - self.patch_size) // self.stride + 1

            y_indices = np.arange(0, num_patches_y * self.stride, self.stride)
            x_indices = np.arange(0, num_patches_x * self.stride, self.stride)
            start_indices = (
                np.array(np.meshgrid(y_indices, x_indices, indexing="ij"))
                .reshape(2, -1)
                .T
            )

            for y, x in start_indices:
                self.patch_indices.append((image_file, y, x))

    def _get_image_shape(self, image_file):
        image = cv2.imread(image_file)
        return image.shape[:2]

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        image_file, y, x = self.patch_indices[idx]
        image = cv2.imread(image_file)
        mask_file = self.mask_files[self.image_files.index(image_file)]
        mask = cv2.imread(mask_file, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_patch = image[y : y + self.patch_size, x : x + self.patch_size]
        mask_patch = mask[y : y + self.patch_size, x : x + self.patch_size]

        masks = [(mask_patch == v["dim"]) for v in self.classes.values()]
        rails = np.bitwise_or.reduce(masks[-2:])
        mask_patch = np.stack([(rails)], axis=-1)

        if self.augmentation:
            augmented = self.augmentation(image=img_patch, mask=mask_patch)
            img_patch, mask_patch = augmented["image"], augmented["mask"]

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img_patch, mask=mask_patch)
            img_patch, mask_patch = preprocessed["image"], preprocessed["mask"]

        return img_patch, mask_patch
