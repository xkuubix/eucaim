import torch
from monai.networks.nets import UNet
from ImagePatcher import ImagePatcher
import numpy as np

class PatchUNet(UNet):
    def __init__(self, config, *args, **kwds):
        super().__init__(*args, **kwds)
        patch_size = config['data'].get('patch_size', 128)
        overlap = config['data'].get('overlap', 0.)
        bag_size = config['data'].get('bag_size', -1)
        empty_thresh = config['data'].get('empty_threshold', 0.)
        self.patcher = ImagePatcher(patch_size=patch_size, overlap=overlap, bag_size=bag_size, empty_thresh=empty_thresh)

    def forward(self, x, mask=None):
        C, H, W = x.shape
        ps = self.patcher.patch_size

        if H < ps or W < ps:
            new_H = max(H, ps)
            new_W = max(W, ps)

            padded_x = torch.zeros((C, new_H, new_W),
                                device=x.device, dtype=x.dtype)
            padded_x[:, :H, :W] = x
            x = padded_x                      # now [1,C,new_H,new_W]

            if mask is not None:
                padded_m = torch.zeros((C, new_H, new_W),
                                    device=mask.device, dtype=mask.dtype)
                padded_m[:, :H, :W] = mask
                mask = padded_m

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
        instances = instances.to(x.device)
        return instances, instances_ids, instances_coords


    # def patch_image_and_mask(self, image, mask):
    #     mask = mask.squeeze(0)
    #     tiles = self.patcher.get_tiles(image.shape[1], image.shape[2])
    #     instances, instances_idx, instances_coords = self.patcher.convert_img_to_bag(image)
    #     mask_instances = []
    #     for patch_coord in instances_coords:
    #         i_id, j_id = patch_coord  # row and col in patch grid
    #         idx = np.where((tiles[:, 4] == i_id) & (tiles[:, 5] == j_id))[0][0]
    #         y, x, h, w = tiles[idx, 0:4].astype(int)  # real image coordinates
    #         patch_mask = mask[y:y+h, x:x+w]           # slice mask for this patch
    #         mask_instances.append(patch_mask)
    #     mask_instances = torch.stack(mask_instances).unsqueeze(1)
    #     return instances, mask_instances, instances_idx, instances_coords


    def patch_image_and_mask(self, image, mask, bg_threshold=0.01, bg_ratio=1.0):
        mask = mask.squeeze(0)
        tiles = self.patcher.get_tiles(image.shape[1], image.shape[2])
        instances, instances_idx, instances_coords = self.patcher.convert_img_to_bag(image)
        mask_instances = []
        for patch_coord in instances_coords:
            i_id, j_id = patch_coord
            idx = np.where((tiles[:, 4] == i_id) & (tiles[:, 5] == j_id))[0][0]
            y, x, h, w = tiles[idx, 0:4].astype(int)
            patch_mask = mask[y:y+h, x:x+w]
            mask_instances.append(patch_mask)
        mask_instances = torch.stack(mask_instances).unsqueeze(1)

        # compute ROI coverage per patch
        roi_frac = mask_instances.flatten(1).mean(dim=1)  # fraction of positive pixels per patch
        pos_idx = torch.where(roi_frac > bg_threshold)[0]
        neg_idx = torch.where(roi_frac <= bg_threshold)[0]

        # random undersampling of negatives
        n_pos = len(pos_idx)
        if self.training:
            n_neg_keep = int(len(neg_idx) if bg_ratio < 0 else bg_ratio * n_pos)
        else:
            n_neg_keep = len(neg_idx)

        if n_neg_keep > 0 and len(neg_idx) > n_neg_keep:
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg_keep]]

        keep_idx = torch.cat([pos_idx, neg_idx])
        instances = instances[keep_idx]
        mask_instances = mask_instances[keep_idx]
        instances_idx = [instances_idx[i] for i in keep_idx]
        instances_coords = [instances_coords[i] for i in keep_idx]

        instances = instances.to(image.device)
        mask_instances = mask_instances.to(mask.device)

        return instances, mask_instances, instances_idx, instances_coords


    @staticmethod
    def norm_instances(instances):
        """
        Normalize instances
        """
        mean = torch.tensor([0.5], device=instances.device).view(1, 1, 1, 1)
        std = torch.tensor([0.5], device=instances.device).view(1, 1, 1, 1)
        return (instances - mean) / std

# %%
def main():
    model = PatchUNet(
        config={"data": {}},
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
        act='PReLU',
        dropout=0.1,
        kernel_size=3,
        up_kernel_size=3,
        norm='INSTANCE',
        bias=False, # using norm
    )
    sample_input = torch.randn(20, 1, 512, 512)
    output = model(sample_input)
    print(output[0])
    print(output[0].shape)
    import matplotlib.pyplot as plt
    plt.imshow(output[0][0].squeeze(0).detach().cpu(),cmap='gray')

if __name__ == '__main__':
    main()

# %%