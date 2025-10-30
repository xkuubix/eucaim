import torch
from monai.networks.nets import UNet
from ImagePatcher import ImagePatcher
import numpy as np

class PatchUNet(UNet):
    def __init__(self, config, *args, **kwds):
        super().__init__(*args, **kwds)
        patch_size = config['data'].get('patch_size', 128)
        overlap = config['data'].get('patch_overlap', 0.)
        bag_size = config['data'].get('bag_size', -1)
        empty_thresh = config['data'].get('empty_thresh', 0.)
        self.patcher = ImagePatcher(patch_size=patch_size, overlap=overlap, bag_size=bag_size, empty_thresh=empty_thresh)
    
    def forward(self, x, mask=None):
        # --- dodać patchowanie ---
        if mask is not None:
            x, mask_patches, instances_ids, _ = self.patch_image_and_mask(x, mask)
        else:
            x, instances_ids, _ = self.patch_image(x)
            mask_patches = None
        # x = self.norm_instances(x) # already in UNet
        pred_patches = super().forward(x)
        return pred_patches, mask_patches, instances_ids 
    
    def patch_image(self, x):
        self.patcher.get_tiles(x.shape[1], x.shape[2])
        instances, instances_ids, instances_coords = self.patcher.convert_img_to_bag(x)
        return instances, instances_ids, instances_coords


    def patch_image_and_mask(self, image, mask):
        mask = mask.squeeze(0)
        tiles = self.patcher.get_tiles(image.shape[1], image.shape[2])
        instances, instances_idx, instances_coords = self.patcher.convert_img_to_bag(image)
        mask_instances = []
        for patch_coord in instances_coords:
            i_id, j_id = patch_coord  # row and col in patch grid
            idx = np.where((tiles[:, 4] == i_id) & (tiles[:, 5] == j_id))[0][0]
            y, x, h, w = tiles[idx, 0:4].astype(int)  # real image coordinates
            patch_mask = mask[y:y+h, x:x+w]           # slice mask for this patch
            mask_instances.append(patch_mask)
        mask_instances = torch.stack(mask_instances).unsqueeze(1)
        return instances, mask_instances, instances_idx, instances_coords

    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances
        """
        mean = torch.tensor([0.5], device=instances.device).view(1, 1, 1, 1)
        std = torch.tensor([0.5], device=instances.device).view(1, 1, 1, 1)
        return (instances - mean) / std


def main():
    model = PatchUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    sample_input = torch.randn(20, 1, 224, 224)
    output = model(sample_input)
    print(output.shape)
    print(output)
    import matplotlib.pyplot as plt
    plt.imshow(output[0][0].detach().cpu(),cmap='gray')

# %%
if __name__ == '__main__':
    main()

# %%