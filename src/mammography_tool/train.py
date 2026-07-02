# %%
from mammography_tool.wandb_utils import fetch_wandb_runs_dataframe
from mammography_tool.models import PatchUNet
from mammography_tool.net_utils import train, test
import gc
import os
import sys
import torch
import torch.distributed as dist
import mammography_tool.utils as utils
import yaml
import wandb
from mammography_tool.ddp_utils import init_distributed, cleanup_distributed

sys.dont_write_bytecode = True
PATH_ = "/users/project1/pt01190/EUCAIM-PG-GUM/code"
if os.getcwd() != PATH_:
    os.chdir(PATH_)

parser = utils.get_args_parser()
args, unknown = parser.parse_known_args()
with open(args.config) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Initialize DDP
is_ddp = False
rank = 0
world_size = 1
local_rank = 0

if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    is_ddp, local_rank, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
else:
    selected_device = config["device"]
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

if config["use_wandb"] and rank == 0:
    os.environ["WANDB_PROJECT"] = "EUC"
    api = wandb.Api()
    entity, project = "jb_pg", "eucaim_cls"
    runs = api.runs(entity + "/" + project)
    run = wandb.init(entity=entity, project=project, config=config)
    run.tags = run.tags + ("dice dla +",)
    run.tags = run.tags + ("undersampling",)
    run.tags = run.tags + ("all patients",)
    run.tags = run.tags + ("val po loss",)
else:
    run = None

if rank == 0:
    print(f"DDP mode: {is_ddp} rank={rank} local_rank={local_rank} world_size={world_size}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Device: {device}")

    # Log SLURM info if available
    if 'SLURM_JOB_ID' in os.environ:
        print(f"SLURM Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"SLURM Nodes: {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}")
        print(f"Master Address: {os.environ.get('MASTER_ADDR', 'N/A')}")
        print(f"Master Port: {os.environ.get('MASTER_PORT', 'N/A')}")

utils.reset_seed(config.get("seed", 42))
dataloaders = utils.get_fold_dataloaders_ddp(config, 0, rank, world_size, is_ddp)

if rank == 0:
    print("Loaded dataloaders")

activation = config.get("activation", "prelu").lower()

model = PatchUNet(
    config,
    num_classes=2,
    mil_hidden=128,
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
    norm="INSTANCE",
    bias=False,  # using norm
).to(device)

# Wrap model with DDP if using distributed training
if is_ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    if rank == 0:
        print("Model wrapped with DistributedDataParallel")

loss_fn_name = config["training_plan"].get("loss_function", "dice")

criterion = utils.get_loss_function(loss_fn_name=loss_fn_name, device=device)

if config["training_plan"].get("optimizer") == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training_plan"]["parameters"]["lr"],
        weight_decay=config["training_plan"]["parameters"]["wd"],
    )
elif config["training_plan"].get("optimizer") == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training_plan"]["parameters"]["lr"],
        weight_decay=config["training_plan"]["parameters"]["wd"],
        momentum=0.9,
    )
else:
    raise ValueError(f"Unsupported optimizer type: {config['training_plan'].get('optimizer')}")

if config.get("resume_from_run"):
    run_id = config["resume_from_run"]

    if rank == 0:
        df = fetch_wandb_runs_dataframe("jb_pg/eucaim_cls")
        run_name = df[df["name"] == run_id]["run_id"].values[0]
        checkpoint_path = os.path.join(config.get("model_path"), f"{run_name}_best.pth")
        print(f"Loading model from wandb run {run_id} at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        resume_payload = {
            "epoch": checkpoint["epoch"],
            "model_state_dict": checkpoint["model"].state_dict()
            if hasattr(checkpoint["model"], "state_dict")
            else checkpoint["model"],
            "optimizer_state_dict": checkpoint["optimizer"].state_dict()
            if hasattr(checkpoint["optimizer"], "state_dict")
            else checkpoint["optimizer"],
        }
    else:
        resume_payload = None

    if is_ddp:
        payload_list = [resume_payload]
        dist.broadcast_object_list(payload_list, src=0)
        resume_payload = payload_list[0]

    epoch = resume_payload["epoch"]
    if is_ddp:
        model.module.load_state_dict(resume_payload["model_state_dict"])
    else:
        model.load_state_dict(resume_payload["model_state_dict"])
    optimizer.load_state_dict(resume_payload["optimizer_state_dict"])


epochs = config["training_plan"]["parameters"].get("epochs", 100)
grad_acc_steps = config["training_plan"]["parameters"].get("grad_acc_steps", 1)
validate_every = config["training_plan"]["parameters"].get("validate_every", 1)
early_stopping_patience = config["training_plan"]["parameters"].get("patience", None)

if rank == 0:
    if run:
        save_path = os.path.join(config.get("model_path"), f"{run.id}_best.pth")
    else:
        save_path = os.path.join(config.get("model_path"), "interactive_best_model.pth")
else:
    save_path = None

history = train(
    model,
    dataloaders,
    optimizer,
    criterion,
    device,
    epochs=epochs,
    grad_acc_steps=grad_acc_steps,
    validate_every=validate_every,
    early_stopping_patience=early_stopping_patience,
    save_path=save_path,
    wandb_run=run,
    is_ddp=is_ddp,
    rank=rank,
)

if rank == 0:
    print("Training finished. History keys:", list(history.keys()))

if history.get("best_model_path"):
    if rank == 0:
        print("Best model saved to:", history["best_model_path"])
        model_path = history["best_model_path"]
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"Loading model from {model_path} [{ckpt['epoch']}]")
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = None

    if is_ddp:
        payload = [state_dict]
        dist.broadcast_object_list(payload, src=0)
        state_dict = payload[0]

    if is_ddp:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

del optimizer
model.eval()
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()

with torch.no_grad():
    test(model, dataloaders["test"], criterion, device, wandb_run=run, rank=rank)

if run:
    run.finish()

if is_ddp:
    cleanup_distributed()

# %%
