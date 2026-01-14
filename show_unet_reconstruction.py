# %%
import os
PATH_ = '/users/project1/pt01190/EUCAIM-PG-GUM/code'
if os.getcwd() != PATH_:
    os.chdir(PATH_)
from models import PatchUNet
import torch
import utils
import yaml
import wandb
import matplotlib.pyplot as plt
import torch
from net_utils import _dice_from_logits_map


parser = utils.get_args_parser()
args, unknown = parser.parse_known_args()
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

selected_device = config['device']
device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

# todo add loading model from run
run = None

dataloaders = utils.get_fold_dataloaders(config, 0)
activation = config.get('activation', 'prelu').lower()

unet = PatchUNet(
    config,
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    num_res_units=2,
    act=activation,
    dropout=0.1,
    kernel_size=3,
    up_kernel_size=3,
    norm='INSTANCE',
    bias=False, # using norm
).to(device)


model_path = "/users/scratch1/jbuler/eucaim/models/MAM-1036_best.pth" # 256
# model_path = "/users/scratch1/jbuler/eucaim/models/MAM-1120_best.pth" # 512
if model_path:
    unet.load_state_dict(torch.load(model_path, map_location=device))

loss_fn_name = config['training_plan'].get('loss_function', 'dice')
criterion = utils.get_loss_function(loss_fn_name=loss_fn_name, device=device)

unet.eval()
i = 0
with torch.no_grad():
    for batch in dataloaders['test']:
        images = batch['image'].to(device)
        masks = batch['annotation'].to(device)
        preds_patched, masks_patched, instances_ids = unet(images, masks)
        pred, patch_count = unet.patcher.reconstruct_image_from_patches(preds_patched, instances_ids, image_shape=images.shape)  # (c, h, w)
        mask, _ = unet.patcher.reconstruct_image_from_patches(masks_patched, instances_ids, image_shape=images.shape) if masks is not None else (None, None)

        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        axs[0].imshow(images[0].cpu(), cmap='gray')
        axs[0].set_title('input')
        axs[1].imshow(mask[0].cpu(), cmap='gray')
        axs[1].set_title('gt')
        probs = torch.sigmoid(pred)
        valid_mask = patch_count > 0
        probs[~valid_mask] = 0.
        preds = (probs > 0.5).float()
        axs[2].imshow(preds[0].cpu(), cmap='gray')
        axs[2].set_title('pred 0.5 th')
        axs[3].imshow(probs[0].cpu(), cmap='hot')
        axs[3].set_title('pred raw')
        
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'Patch Size: {config["data"]["patch_size"]} (n={preds_patched.shape[0]}), Overlap: {config["data"]["overlap"]} | Image: {images.shape[-2]}x{images.shape[-1]}px', fontsize=16)
        plt.tight_layout()
        plt.show()
        i += 1
        if i == 10:
            break
# %%