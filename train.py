# %%
import os
PATH_ = '/users/project1/pt01190/EUCAIM-PG-GUM/code'
if os.getcwd() != PATH_:
    os.chdir(PATH_)
from models import PatchUNet
import torch
import utils
import yaml
from losses import DiceFocalLoss
from monai.losses import GeneralizedDiceFocalLoss
from net_utils import train, test
import neptune


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

criterion = GeneralizedDiceFocalLoss(
    sigmoid=True,
    to_onehot_y=False,
    reduction="mean",
    lambda_gdl=1.0,
    lambda_focal=1.0,
    gamma=4.0,
).to(device)

if config['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(unet.parameters(), lr=config['training_plan']['parameters']['lr'])
elif config['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(unet.parameters(), lr=config['training_plan']['parameters']['lr'], momentum=0.9)
else:
    raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")

epochs = config['training_plan']['parameters'].get('epochs', 100)
validate_every = config['training_plan']['parameters'].get('validate_every', 1)
early_stopping_patience = config['training_plan']['parameters'].get('patience', None)
checkpoint_path = config.get('model_path') + run['sys/id'].fetch() + '_best.pth'

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
)

print('Training finished. History keys:', list(history.keys()))
if history.get('best_model_path'):
    print('Best model saved to:', history['best_model_path'])

# Run final test
test_stats = test(unet, dataloaders['test'], criterion, device)
print('Test results:', test_stats)

# %%

