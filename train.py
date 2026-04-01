# %%
import os
import re
PATH_ = '/users/project1/pt01190/EUCAIM-PG-GUM/code'
if os.getcwd() != PATH_:
    os.chdir(PATH_)
from models import PatchUNet
import torch
import utils
import yaml
from net_utils import train, test
import wandb


parser = utils.get_args_parser()
args, unknown = parser.parse_known_args()
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

selected_device = config['device']
device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

if config["use_wandb"]:
    os.environ["WANDB_PROJECT"] = "EUC"
    api = wandb.Api()
    entity, project = "jb_pg", "eucaim"
    runs = api.runs(entity + "/" + project)
    run = wandb.init(entity=entity, project=project, config=config)
    run.tags = run.tags + ("SEG",)
    run.tags = run.tags + ("CLEAR-AI",)
    run.tags = run.tags + ("dice dla +",)
    run.tags = run.tags + ("undersampling",)
    run.tags = run.tags + ("positive patients",)
else:
    run = None
utils.reset_seed(config.get('seed', 42))
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

loss_fn_name = config['training_plan'].get('loss_function', 'dice')

criterion = utils.get_loss_function(loss_fn_name=loss_fn_name, device=device)


if config['training_plan'].get('optimizer') == 'adam':
    optimizer = torch.optim.Adam(unet.parameters(), lr=config['training_plan']['parameters']['lr'])
elif config['training_plan'].get('optimizer') == 'sgd':
    optimizer = torch.optim.SGD(unet.parameters(), lr=config['training_plan']['parameters']['lr'], momentum=0.9)
else:
    raise ValueError(f"Unsupported optimizer type: {config['training_plan'].get('optimizer')}")

epochs = config['training_plan']['parameters'].get('epochs', 100)
validate_every = config['training_plan']['parameters'].get('validate_every', 1)
early_stopping_patience = config['training_plan']['parameters'].get('patience', None)
if run:
    checkpoint_path = os.path.join(config.get('model_path'), f"{run.id}_best.pth")
else:
    checkpoint_path = os.path.join(config.get('model_path'), "interactive_best_model.pth")

history = train(
    unet,
    dataloaders,
    optimizer,
    criterion,
    device,
    epochs=epochs,
    validate_every=validate_every,
    early_stopping_patience=early_stopping_patience,
    save_path=checkpoint_path,
    wandb_run=run
)

print('Training finished. History keys:', list(history.keys()))
if history.get('best_model_path'):
    print('Best model saved to:', history['best_model_path'])

# Run final test
test(unet, dataloaders['test'], criterion, device, wandb_run=run)

if run:
    run.finish()
# %%

