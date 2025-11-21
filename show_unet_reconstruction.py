# %%
import os
PATH_ = '/users/project1/pt01190/EUCAIM-PG-GUM/code'
if os.getcwd() != PATH_:
    os.chdir(PATH_)
from models import PatchUNet
import torch
import utils
import yaml
import neptune
import matplotlib.pyplot as plt
import torch
from net_utils import _dice_from_logits_map

parser = utils.get_args_parser()
args, unknown = parser.parse_known_args()
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

selected_device = config['device']
device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

if config["neptune"]:
    run = neptune.init_run(project="ProjektMMG/Mammografia")
    run["sys/group_tags"].add(["SEG"])
    run["sys/group_tags"].add(["CLEAR-AI"])
    run["sys/group_tags"].add(["dice dla +"])
    run["sys/group_tags"].add(["weighted CE"])
    run["sys/tags"].add(["undersampling"])
    run["sys/tags"].add(["positive patients"])
    run["config"] = config
else:
    run = None

dataloaders = utils.get_fold_dataloaders(config, 0)

unet = PatchUNet(
    config,
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
).to(device)

loss_fn_name = config['training_plan'].get('loss_function', 'dice')
criterion = utils.get_loss_function(loss_fn_name=loss_fn_name, device=device)

unet.eval()
with torch.no_grad():
    for batch in dataloaders['val']:
        images = batch['image'].to(device)
        masks = batch['annotation'].to(device)

        preds_patched, masks_patched, instances_ids = unet(images, masks)
        print(preds_patched.shape, masks_patched.shape)
        pred, patch_count = unet.patcher.reconstruct_image_from_patches(preds_patched, instances_ids, image_shape=images.shape)  # (c, h, w)
        mask, _ = unet.patcher.reconstruct_image_from_patches(masks_patched, instances_ids, image_shape=images.shape) if masks is not None else (None, None)
        print(pred.shape, mask.shape)
        loss = criterion(preds_patched, masks_patched)
        print(_dice_from_logits_map(pred, mask, 0.9, patch_count))
        print(float(loss.item()))
        probs = torch.sigmoid(pred)
        valid_mask = patch_count > 0
        probs[~valid_mask] = 0.
        preds = (probs > 0.5).float()
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(images[0].cpu(), cmap='gray')
        axs[0].set_title('Input Image')
        axs[1].imshow(mask[0].cpu(), cmap='gray')
        axs[1].set_title('Ground Truth Mask')
        axs[2].imshow(probs[0].cpu(), cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[3].imshow(preds[0].cpu(), cmap='gray')
        axs[3].set_title('Binary Prediction')
        plt.show()
        break

# %%