import os
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.distributed as dist
from typing import Dict, Tuple, Optional, List, Any
from mammography_tool.wandb_utils import _safe_wandb_log
import time
from functools import wraps


def timeit(unit="sec"):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            elapsed = (time.time() - start) / (60 if unit == "min" else 1)
            rank = kwargs.get("rank", 0)
            if rank == 0:
                print(f"{fn.__name__} took {elapsed:.1f} {unit}")
            return result

        return wrapper

    return decorator


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _pixel_auprc(logits: torch.Tensor, targets: torch.Tensor, patch_count: torch.Tensor = None) -> float:
    with torch.no_grad():
        probs = torch.sigmoid(logits)

        if patch_count is not None:
            mask_valid = patch_count > 0
            probs = probs[mask_valid]
            targets = targets[mask_valid]

        probs_np = probs.cpu().numpy().ravel()
        targets_np = targets.cpu().numpy().ravel().astype(int)

        if targets_np.sum() == 0:  # no positive pixels
            return float("nan")

        return average_precision_score(targets_np, probs_np)


def _binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, int]:
    """
    Compute Dice score for a batch given raw logits (not probabilities).

    Returns:
        dice_mean: mean dice over the batch (scalar)
        n_items: number of items used to compute the mean
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        # sum over spatial dims
        dims = tuple(range(1, preds.dim()))
        tp = (preds * targets).sum(dim=dims)
        fp = (preds * (1 - targets)).sum(dim=dims)
        fn = ((1 - preds) * targets).sum(dim=dims)

        # add eps to denom
        denom = 2 * tp + fp + fn + 1e-7
        dice_per_item = 2 * tp / denom

        # exclude fully negative patches (no ROI in target and prediction)
        valid_mask = (targets.sum(dim=dims) + preds.sum(dim=dims)) > 0
        valid_dice = dice_per_item[valid_mask]

        if valid_dice.numel() == 0:
            return 1.0, 0  # if all patches are empty, define Dice = 1.0 or return n_items = 0

        dice_mean = valid_dice.mean().item()
        n_items = valid_dice.numel()

    return float(dice_mean), int(n_items)


def _dice_from_logits_map(
    logits: torch.Tensor, targets: torch.Tensor, patch_count: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Compute Dice score for a batch of reconstructed full-image logits.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        mask_valid = patch_count > 0

        probs = probs[mask_valid]
        targets = targets[mask_valid]

        preds = (probs > threshold).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        return (2 * tp / (2 * tp + fp + fn + 1e-7)).item()


def classification_metrics(logits, targets):
    """
    Args:
        logits:  list of [1, num_classes] tensors
        targets: list of int labels
    """
    logits = torch.cat(logits, dim=0)  # [N, C]
    probs = logits.softmax(dim=-1)[:, 1]  # positive class prob
    preds = logits.argmax(dim=-1)
    targets = torch.tensor(targets)

    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.numpy()
    # preds_np = preds.detach().cpu().numpy()

    acc = (preds == targets).float().mean().item()
    auroc = roc_auc_score(targets_np, probs_np)
    auprc = average_precision_score(targets_np, probs_np)  # better than AUROC for imbalanced

    return {"acc": acc, "auroc": auroc, "auprc": auprc}


def _get_pred_rates_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """
    Compute positive and negative prediction rates for a batch given raw logits (not probabilities).

    Returns:
        fpr: false positive rate (predicted positive but actually negative)
        fnr: false negative rate (predicted negative but actually positive)
        tpr: true positive rate (predicted positive and actually positive)
        tnr: true negative rate (predicted negative and actually negative)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()

        fpr = fp / (fp + tn + 1e-7)
        fnr = fn / (tp + fn + 1e-7)
        tpr = tp / (tp + fn + 1e-7)
        tnr = tn / (fp + tn + 1e-7)
    return fpr, fnr, tpr, tnr


@timeit(unit="min")
def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=None, grad_acc_steps=1):
    model.train()
    running = {"seg_loss": 0.0, "cls_loss": 0.0, "dice": 0.0}
    n_pos = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["annotation"].to(device)
        labels = batch["patientclass"].squeeze(0).to(device).long()

        preds_patched, masks_patched, _, cls_logits, attn, seg_mask = model(images, masks, bg_threshold=0.01, bg_ratio=2.0)

        # --- segmentation loss (positive cases only) ---
        if labels.item() > 0 and seg_mask is not None and seg_mask.any():
            selected_preds = preds_patched[seg_mask]
            selected_masks = masks_patched[seg_mask]
            seg_loss = criterion(selected_preds, selected_masks)

            with torch.no_grad():
                has_fg = selected_masks.flatten(1).sum(dim=1) > 0
                if has_fg.any():
                    dice_mean, _ = _binary_metrics_from_logits(selected_preds[has_fg], selected_masks[has_fg])
                    running["dice"] += dice_mean
                    n_pos += 1
        else:
            seg_loss = (preds_patched * 0).sum()  # has grad_fn, touches model params

        # --- classification loss ---
        cls_loss = torch.nn.functional.cross_entropy(cls_logits, labels)

        running["seg_loss"] += seg_loss.item()
        running["cls_loss"] += cls_loss.item()

        total_loss = (seg_loss + cls_loss * 0.5) / grad_acc_steps
        total_loss.backward()

        if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(dataloader):
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    n_batches = len(dataloader)
    return {
        "seg_loss": running["seg_loss"] / n_batches,
        "cls_loss": running["cls_loss"] / n_batches,
        "dice": running["dice"] / max(1, n_pos),
    }


@timeit(unit="min")
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    is_ddp: bool = False,
    rank: int = 0,
) -> Dict[str, float]:
    """
    Run validation (no grad). Returns average loss and dice.
    """
    model.eval()
    running = {"seg_loss": 0.0, "cls_loss": 0.0, "dice": 0.0, "auprc": 0.0, "lesion_detected": 0.0}
    n_pos = 0
    n_patches = 0
    all_preds: List[Tuple[object, torch.Tensor]] = []
    all_labels: List[torch.Tensor] = []
    base_model = _unwrap_model(model)
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["annotation"].to(device)
            labels = batch["patientclass"].squeeze(0).to(device)

            preds_patched, masks_patched, instances_ids, cls_logits, attn, seg_mask = model(
                images, masks, bg_threshold=0.01, bg_ratio=2.0
            )
            all_preds.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
            cls_loss = torch.nn.functional.cross_entropy(cls_logits, labels)
            running["cls_loss"] += float(cls_loss)
            if labels.item() > 0 and seg_mask is not None and seg_mask.any():
                preds_cpu = preds_patched.cpu()
                masks_cpu = masks_patched.cpu()
                seg_loss = criterion(preds_patched[seg_mask], masks_patched[seg_mask])
                n_patches += seg_mask.sum().item()
                running["seg_loss"] += float(seg_loss.item()) * seg_mask.sum().item()  # weighted by number of patches
                pred, patch_count = base_model.patcher.reconstruct_image_from_patches(
                    preds_cpu, instances_ids, image_shape=images.shape
                )
                mask_reconstructed, _ = base_model.patcher.reconstruct_image_from_patches(
                    masks_cpu, instances_ids, image_shape=images.shape
                )
                running["dice"] += _dice_from_logits_map(pred, mask_reconstructed, patch_count=patch_count)
                running["auprc"] += _pixel_auprc(pred, mask_reconstructed, patch_count=patch_count)
                probs = torch.sigmoid(pred)
                pred_binary = (probs > 0.5).float()
                valid = patch_count > 0
                pred_binary[~valid] = 0.0
                tp_image = (pred_binary * mask_reconstructed).sum(dim=(-2, -1))
                running["lesion_detected"] += (tp_image > 0).float().mean().item()

                n_pos += 1
                del pred, mask_reconstructed, preds_cpu, masks_cpu, pred_binary

            del preds_patched, masks_patched, images, masks, cls_logits, attn
            torch.cuda.empty_cache()

    local_payload = {
        "seg_loss": running["seg_loss"],
        "cls_loss": running["cls_loss"],
        "dice": running["dice"],
        "auprc": running["auprc"],
        "lesion_detected": running["lesion_detected"],
        "n_pos": n_pos,
        "n_patches": n_patches,
        "n_batches": len(dataloader),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }

    if is_ddp and dist.is_initialized():
        gathered_payloads = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_payloads, local_payload)
        if rank == 0:
            all_preds = []
            all_labels = []
            total_seg_loss = 0.0
            total_cls_loss = 0.0
            total_dice = 0.0
            total_auprc = 0.0
            total_lesion_detected = 0.0
            total_n_pos = 0
            total_n_patches = 0
            total_n_batches = 0
            for payload in gathered_payloads:
                total_seg_loss += payload["seg_loss"]
                total_cls_loss += payload["cls_loss"]
                total_dice += payload["dice"]
                total_auprc += payload["auprc"]
                total_lesion_detected += payload["lesion_detected"]
                total_n_pos += payload["n_pos"]
                total_n_patches += payload["n_patches"]
                total_n_batches += payload["n_batches"]
                all_preds.extend(payload["all_preds"])
                all_labels.extend(payload["all_labels"])

            cls_metrics = classification_metrics(all_preds, torch.cat(all_labels).numpy())
            final_stats = {
                # "seg_loss": total_seg_loss / max(1, total_n_pos),
                "seg_loss": total_seg_loss / max(1, total_n_patches),
                "cls_loss": total_cls_loss / max(1, total_n_batches),
                "dice": total_dice / max(1, total_n_pos),
                "auprc": total_auprc / max(1, total_n_pos),
                "lesion_detected": total_lesion_detected / max(1, total_n_pos),
                "cls_metrics": cls_metrics,
            }
        else:
            final_stats = None

        final_stats_list = [final_stats]
        dist.broadcast_object_list(final_stats_list, src=0)
        return final_stats_list[0]

    avg_seg_loss = running["seg_loss"] / max(1, n_pos)  # average seg loss only over positive cases
    avg_cls_loss = running["cls_loss"] / max(1, len(dataloader))
    avg_dice = running["dice"] / max(1, n_pos)  # average dice only over positive cases
    avg_auprc = running["auprc"] / max(1, n_pos)  # average auprc only over positive cases
    cls_metrics = classification_metrics(all_preds, torch.cat(all_labels).numpy())

    return {
        "seg_loss": avg_seg_loss,
        "cls_loss": avg_cls_loss,
        "dice": avg_dice,
        "auprc": avg_auprc,
        "lesion_detected": running["lesion_detected"] / max(1, n_pos),
        "cls_metrics": cls_metrics,
    }


@timeit(unit="min")
def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    rank: int = 0,
    is_ddp: bool = False,
) -> Dict[str, object]:
    """
    Run a test loop. If `criterion` is provided, compute loss as well.

    Args:
        model: PyTorch model (may be wrapped with DistributedDataParallel)
        dataloader: Test DataLoader
        criterion: Loss function
        device: Device to test on
        wandb_run: Optional wandb run object (only logs on rank 0)
        rank: Current process rank (0 is primary)

    If `return_predictions` is True, returns a list of (instances_ids, probs) per batch.
    """
    model.eval()
    running = {"seg_loss": 0.0, "cls_loss": 0.0, "dice": 0.0, "auprc": 0.0}
    running["px_fpr"] = 0.0
    running["px_fnr"] = 0.0
    running["px_tpr"] = 0.0
    running["px_tnr"] = 0.0
    n_pos = 0
    n_neg = 0
    all_preds: List[Tuple[object, torch.Tensor]] = []
    all_labels: List[torch.Tensor] = []
    base_model = _unwrap_model(model)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch.get("annotation")
            labels = batch.get("patientclass").squeeze(0).to(device) if batch.get("patientclass") is not None else None
            if masks is not None:
                masks = masks.to(device)

            preds_patched, masks_patched, instances_ids, cls_logits, _, _ = model(images, masks)
            all_preds.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
            if labels == 1:
                n_pos += 1
                pred, patch_count = base_model.patcher.reconstruct_image_from_patches(
                    preds_patched, instances_ids, image_shape=images.shape
                )  # (c, h, w)
                mask_reconstructed, _ = (
                    base_model.patcher.reconstruct_image_from_patches(masks_patched, instances_ids, image_shape=images.shape)
                    if masks is not None
                    else None
                )
                seg_loss = criterion(pred, mask_reconstructed) if mask_reconstructed is not None else 0.0
                running["dice"] += _dice_from_logits_map(pred, mask_reconstructed, threshold=0.5, patch_count=patch_count)
                running["auprc"] += _pixel_auprc(pred, mask_reconstructed, patch_count=patch_count)
                fpr, fnr, tpr, tnr = _get_pred_rates_from_logits(pred, mask_reconstructed)
                running["px_fpr"] += fpr
                running["px_fnr"] += fnr
                running["px_tpr"] += tpr
                running["px_tnr"] += tnr
            else:
                seg_loss = torch.tensor(0.0, device=device, requires_grad=False)
                n_neg += 1
            cls_loss = torch.nn.functional.cross_entropy(cls_logits, labels.to(device))

            running["seg_loss"] += float(seg_loss) if isinstance(seg_loss, float) else float(seg_loss.item())
            running["cls_loss"] += float(cls_loss.item())
            del images, masks, preds_patched, masks_patched
            torch.cuda.empty_cache()

    local_payload = {
        "seg_loss": running["seg_loss"],
        "cls_loss": running["cls_loss"],
        "dice": running["dice"],
        "auprc": running["auprc"],
        "px_fpr": running["px_fpr"],
        "px_fnr": running["px_fnr"],
        "px_tpr": running["px_tpr"],
        "px_tnr": running["px_tnr"],
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_batches": len(dataloader),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }

    if is_ddp and dist.is_initialized():
        gathered_payloads = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_payloads, local_payload)
        if rank == 0:
            all_preds = []
            all_labels = []
            totals = {
                "seg_loss": 0.0,
                "cls_loss": 0.0,
                "dice": 0.0,
                "auprc": 0.0,
                "px_fpr": 0.0,
                "px_fnr": 0.0,
                "px_tpr": 0.0,
                "px_tnr": 0.0,
            }
            total_n_pos = 0
            total_n_neg = 0
            total_n_batches = 0
            for payload in gathered_payloads:
                for key in totals:
                    totals[key] += payload[key]
                total_n_pos += payload["n_pos"]
                total_n_neg += payload["n_neg"]
                total_n_batches += payload["n_batches"]
                all_preds.extend(payload["all_preds"])
                all_labels.extend(payload["all_labels"])

            cls_metrics = classification_metrics(all_preds, torch.cat(all_labels).numpy())
            final_stats = {
                "seg_loss": totals["seg_loss"] / max(1, total_n_pos),
                "cls_loss": totals["cls_loss"] / max(1, total_n_batches),
                "dice": totals["dice"] / max(1, total_n_pos),
                "auprc": totals["auprc"] / max(1, total_n_pos),
                "px_fpr": totals["px_fpr"] / max(1, total_n_pos),
                "px_fnr": totals["px_fnr"] / max(1, total_n_pos),
                "px_tpr": totals["px_tpr"] / max(1, total_n_pos),
                "px_tnr": totals["px_tnr"] / max(1, total_n_neg),
                "cls_metrics": cls_metrics,
            }
        else:
            final_stats = None

        final_stats_list = [final_stats]
        dist.broadcast_object_list(final_stats_list, src=0)
        final_stats = final_stats_list[0]
    else:
        cls_metrics = classification_metrics(all_preds, torch.cat(all_labels).numpy())
        final_stats = {
            "seg_loss": running["seg_loss"] / max(1, n_pos),
            "cls_loss": running["cls_loss"] / max(1, len(dataloader.dataset)),
            "dice": running["dice"] / max(1, n_pos),
            "auprc": running["auprc"] / max(1, n_pos),
            "px_fpr": running["px_fpr"] / max(1, n_pos),
            "px_fnr": running["px_fnr"] / max(1, n_pos),
            "px_tpr": running["px_tpr"] / max(1, n_pos),
            "px_tnr": running["px_tnr"] / max(1, n_neg),
            "cls_metrics": cls_metrics,
        }

    if wandb_run is not None and rank == 0:
        wandb_run.summary["test/seg/loss"] = final_stats["seg_loss"]
        wandb_run.summary["test/cls/loss"] = final_stats["cls_loss"]
        wandb_run.summary["test/seg/dice"] = final_stats["dice"]
        wandb_run.summary["test/seg/auprc"] = final_stats["auprc"]
        wandb_run.summary["test/cls/acc"] = final_stats["cls_metrics"]["acc"]
        wandb_run.summary["test/cls/auroc"] = final_stats["cls_metrics"]["auroc"]
        wandb_run.summary["test/cls/auprc"] = final_stats["cls_metrics"]["auprc"]
        wandb_run.summary["test/seg/px_fpr"] = final_stats["px_fpr"]
        wandb_run.summary["test/seg/px_fnr"] = final_stats["px_fnr"]
        wandb_run.summary["test/seg/px_tpr"] = final_stats["px_tpr"]
        wandb_run.summary["test/seg/px_tnr"] = final_stats["px_tnr"]

    return final_stats


@timeit(unit="min")
def train(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
    scheduler: Optional[object] = None,
    clip_grad: Optional[float] = None,
    grad_acc_steps: int = 1,
    validate_every: int = 1,
    early_stopping_patience: Optional[int] = None,
    save_path: Optional[str] = None,
    min_delta: float = 1e-8,
    wandb_run: Optional[Any] = None,
    is_ddp: bool = False,
    rank: int = 0,
) -> Dict[str, object]:
    """
    High-level training loop with DDP support.

    Args:
        model: PyTorch model (may be wrapped with DistributedDataParallel)
        dataloaders: Dictionary of DataLoaders for train/val/cal/test
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs to train
        scheduler: Optional learning rate scheduler
        clip_grad: Optional gradient clipping value
        grad_acc_steps: Gradient accumulation steps
        validate_every: Validate every N epochs
        early_stopping_patience: Early stopping patience (if None, disabled)
        save_path: Path to save best model (only used on rank 0)
        min_delta: Minimum improvement threshold for early stopping
        wandb_run: Optional wandb run object (only logs on rank 0)
        is_ddp: Whether using DistributedDataParallel
        rank: Current process rank (0 is primary)

    Returns:
        history dict with lists for 'train_loss','train_dice','val_loss','val_dice' (when validation available).
    """
    history = {"train_seg_loss": [], "train_cls_loss": [], "train_dice": [], "train_auprc": []}
    if "val" in dataloaders:
        history.update({"val_seg_loss": [], "val_cls_loss": [], "val_dice": [], "val_auprc": [], "val_detection_rate": []})

    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    best_model_path: Optional[str] = None
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        if is_ddp:
            dataloaders["train"].sampler.set_epoch(epoch)

        train_stats = train_epoch(model, dataloaders["train"], optimizer, criterion, device, clip_grad, grad_acc_steps)
        history["train_seg_loss"].append(train_stats["seg_loss"])
        history["train_cls_loss"].append(train_stats["cls_loss"])
        history["train_dice"].append(train_stats["dice"])

        if wandb_run is not None and rank == 0:
            _safe_wandb_log(wandb_run, "train/seg/loss", train_stats["seg_loss"], step=epoch)
            _safe_wandb_log(wandb_run, "train/cls/loss", train_stats["cls_loss"], step=epoch)
            _safe_wandb_log(wandb_run, "train/seg/dice", train_stats["dice"], step=epoch)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        did_validate = False
        if "val" in dataloaders and ((epoch + 1) % validate_every == 0):
            val_stats = validate(model, dataloaders["val"], criterion, device, is_ddp=is_ddp, rank=rank)
            history["val_seg_loss"].append(val_stats["seg_loss"])
            history["val_cls_loss"].append(val_stats["cls_loss"])
            history["val_dice"].append(val_stats["dice"])
            history["val_auprc"].append(val_stats["auprc"])
            history["val_detection_rate"].append(val_stats["lesion_detected"])

            if wandb_run is not None and rank == 0:
                _safe_wandb_log(wandb_run, "val/seg/loss", val_stats["seg_loss"], step=epoch)
                _safe_wandb_log(wandb_run, "val/cls/loss", val_stats["cls_loss"], step=epoch)
                _safe_wandb_log(wandb_run, "val/seg/dice", val_stats["dice"], step=epoch)
                _safe_wandb_log(wandb_run, "val/seg/auprc", val_stats["auprc"], step=epoch)
                _safe_wandb_log(wandb_run, "val/seg/lesion_detected", val_stats["lesion_detected"], step=epoch)
                _safe_wandb_log(wandb_run, "val/cls/acc", val_stats["cls_metrics"]["acc"], step=epoch)
                _safe_wandb_log(wandb_run, "val/cls/auroc", val_stats["cls_metrics"]["auroc"], step=epoch)
                _safe_wandb_log(wandb_run, "val/cls/auprc", val_stats["cls_metrics"]["auprc"], step=epoch)
            did_validate = True

            if early_stopping_patience is not None:
                current_val = val_stats["seg_loss"]  # or use val_stats['dice'] with reversed logic
                # improvement if decrease less than min_delta
                should_stop = False
                if current_val < best_val - min_delta:
                    best_val = current_val
                    best_epoch = epoch
                    epochs_since_improve = 0
                    # save best model (only on rank 0)
                    if save_path is not None and rank == 0:
                        checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": _unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                        torch.save(checkpoint, save_path)
                        best_model_path = save_path
                        print(f"[Rank {rank}] Saved best model (val loss {best_val:.6f}) to {save_path}")
                        if wandb_run is not None:
                            _safe_wandb_log(wandb_run, "best/model_path", save_path, step=epoch)
                            _safe_wandb_log(wandb_run, "best/val_loss", best_val, step=epoch)
                            _safe_wandb_log(
                                wandb_run,
                                "early_stopping/patience",
                                early_stopping_patience - epochs_since_improve,
                                step=epoch,
                            )
                else:
                    epochs_since_improve += 1
                    if wandb_run is not None and rank == 0:
                        _safe_wandb_log(
                            wandb_run, "early_stopping/patience", early_stopping_patience - epochs_since_improve, step=epoch
                        )

                if epochs_since_improve >= early_stopping_patience:
                    should_stop = True
                    if rank == 0:
                        print(
                            f"Early stopping triggered at epoch {epoch+1}. No improvement for {epochs_since_improve} validation checks."
                        )
                        if wandb_run is not None:
                            _safe_wandb_log(wandb_run, "early_stopping/stopped_epoch", epoch + 1, step=epoch)
                            _safe_wandb_log(
                                wandb_run,
                                "early_stopping/patience",
                                early_stopping_patience - epochs_since_improve,
                                step=epoch,
                            )
                if is_ddp and dist.is_initialized():
                    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())
                if should_stop:
                    break

        if rank == 0:
            print(
                f"[Rank {rank}] Epoch {epoch+1}/{epochs} | TRAIN SL: {train_stats['seg_loss']:.4f} CL: {train_stats['cls_loss']:.4f} D: {train_stats['dice']:.4f}",
                end="",
            )
            if did_validate:
                print(
                    f" | VAL SL: {val_stats['seg_loss']:.4f} CL: {val_stats['cls_loss']:.4f} D: {val_stats['dice']:.4f}, Detected: {val_stats['lesion_detected']:.4f}"
                )
            else:
                print("")

    history_out: Dict[str, object] = history
    history_out["best_epoch"] = best_epoch
    history_out["best_val_loss"] = best_val if best_epoch >= 0 else None
    history_out["best_model_path"] = best_model_path

    if is_ddp and dist.is_initialized():
        path_list = [best_model_path]
        dist.broadcast_object_list(path_list, src=0)
        best_model_path = path_list[0]

    history_out["best_model_path"] = best_model_path

    if wandb_run is not None:
        if best_model_path is not None:
            wandb_run.summary["best/model_path"] = best_model_path
            wandb_run.summary["best/val_loss"] = history_out["best_val_loss"]
            wandb_run.summary["best/epoch"] = best_epoch

    return history_out
