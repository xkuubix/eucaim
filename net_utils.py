import os
import torch
from typing import Dict, Tuple, Optional, List, Any
from wandb_utils import _safe_wandb_log

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
        denom = (2 * tp + fp + fn + 1e-7)
        dice_per_item = 2 * tp / denom

        # exclude fully negative patches (no ROI in target and prediction)
        valid_mask = (targets.sum(dim=dims) + preds.sum(dim=dims)) > 0
        valid_dice = dice_per_item[valid_mask]

        if valid_dice.numel() == 0:
            return 1.0, 0  # if all patches are empty, define Dice = 1.0 or return n_items = 0

        dice_mean = valid_dice.mean().item()
        n_items = valid_dice.numel()

    return float(dice_mean), int(n_items)


def _dice_from_logits_map(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, patch_count: Optional[torch.Tensor] = None) -> float:
    """
    Compute Dice score for a batch of reconstructed full-image logits.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        mask_valid = patch_count > 0
        probs[~mask_valid] = 0.
        preds = (probs > threshold).float()
        dims = tuple(range(1, preds.dim()))  # sum over spatial dims
        tp = (preds * targets).sum(dim=dims)
        fp = (preds * (1 - targets)).sum(dim=dims)
        fn = ((1 - preds) * targets).sum(dim=dims)

        dice_per_image = 2 * tp / (2 * tp + fp + fn + 1e-7)
        return dice_per_image.mean().item()



def train_epoch(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        clip_grad: Optional[float] = None,
        ) -> Dict[str, float]:
    """
    Run one training epoch.

    Returns a dict with average loss and average dice.
    """
    model.train()
    running = { "seg_loss": 0.0, "cls_loss": 0.0, "dice": 0.0 }
    n_pos = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['annotation'].to(device)
        labels = batch['patientclass'].squeeze(0).to(device)

        optimizer.zero_grad()
        # TODO: guide attn as neg undersample (instead of seg_mask)?
        preds_patched, masks_patched, _, cls_logits, attn, seg_mask = model(images, masks, bg_ratio=-1)

        if seg_mask.any():
            seg_loss = criterion(preds_patched[seg_mask], masks_patched[seg_mask])
            dice_mean, _ = _binary_metrics_from_logits(preds_patched[seg_mask], masks_patched[seg_mask])
            running["dice"] += dice_mean
            n_pos += 1
        else:
            seg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        cls_loss = torch.nn.functional.cross_entropy(cls_logits, labels.to(device))
        running["cls_loss"] += float(cls_loss.item())
        running["seg_loss"] += float(seg_loss.item())
        loss = seg_loss + cls_loss * 0.3
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    avg_dice = running["dice"] / max(1, n_pos)
    avg_seg_loss = running["seg_loss"] / max(1, n_pos)
    avg_cls_loss = running["cls_loss"] / max(1, len(dataloader))
    return {"dice": avg_dice, "seg_loss": avg_seg_loss, "cls_loss": avg_cls_loss}


def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    ) -> Dict[str, float]:
    """
    Run validation (no grad). Returns average loss and dice.
    """
    model.eval()
    running = {"seg_loss": 0.0, "cls_loss": 0.0, "dice": 0.0, "lesion_detected": 0.0}
    n_pos = 0
    all_preds: List[Tuple[object, torch.Tensor]] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['annotation'].to(device)
            labels = batch['patientclass'].squeeze(0).to(device)

            preds_patched, masks_patched, instances_ids, cls_logits, attn, seg_mask = model(images, masks)
            all_preds.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
            if seg_mask.any():
                seg_loss = criterion(preds_patched[seg_mask], masks_patched[seg_mask])
                pred, patch_count = model.patcher.reconstruct_image_from_patches(preds_patched, instances_ids, image_shape=images.shape)  # (c, h, w)
                mask_reconstructed, _ = model.patcher.reconstruct_image_from_patches(masks_patched, instances_ids, image_shape=images.shape)
                dice_mean = _dice_from_logits_map(pred, mask_reconstructed, patch_count=patch_count)
                running["dice"] += dice_mean
                tp_image = (pred * mask_reconstructed).sum(dim=(-2,-1))
                lesion_detected = (tp_image > 0).float().mean().item()
                running["lesion_detected"] += lesion_detected
                n_pos += 1
            else:
                seg_loss = torch.tensor(0.0, device=device, requires_grad=False)

            cls_loss = torch.nn.functional.cross_entropy(cls_logits, labels.to(device))
            running["seg_loss"] += float(seg_loss)
            running["cls_loss"] += float(cls_loss)

    avg_seg_loss    = running["seg_loss"] / max(1, n_pos)  # average seg loss only over positive cases
    avg_cls_loss    = running["cls_loss"] / max(1, len(dataloader))
    avg_dice        = running["dice"] / max(1, n_pos)  # average dice only over positive cases
    cls_metrics = classification_metrics(all_preds, torch.cat(all_labels).numpy())
    
    return {
        "seg_loss": avg_seg_loss,
        "cls_loss": avg_cls_loss,
        "dice":     avg_dice,
        "lesion_detected": running["lesion_detected"] / max(1, n_pos),
        "cls_metrics": cls_metrics
        }


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    return_predictions: bool = False,
    wandb_run: Optional[Any] = None,
) -> Dict[str, object]:
    """
    Run a test loop. If `criterion` is provided, compute loss as well.

    If `return_predictions` is True, returns a list of (instances_ids, probs) per batch.
    """
    model.eval()
    running = {"seg_loss": 0.0, "cls_loss": 0.0, "dice": 0.0}
    running['px_fpr'] = 0.0
    running['px_fnr'] = 0.0
    running['px_tpr'] = 0.0
    running['px_tnr'] = 0.0
    n_pos = 0
    n_neg = 0
    all_preds: List[Tuple[object, torch.Tensor]] = []
    all_labels: List[torch.Tensor] = []
    all_preds: List[Tuple[object, torch.Tensor]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch.get('annotation')
            labels = batch.get('patientclass').squeeze(0).to(device) if batch.get('patientclass') is not None else None
            if masks is not None:
                masks = masks.to(device)

            preds_patched, masks_patched, instances_ids, cls_logits, attn, seg_mask = model(images, masks)
            all_preds.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
            if labels == 1:
                n_pos += 1
                pred, patch_count = model.patcher.reconstruct_image_from_patches(preds_patched, instances_ids, image_shape=images.shape)  # (c, h, w)
                mask_reconstructed, _ = model.patcher.reconstruct_image_from_patches(masks_patched, instances_ids, image_shape=images.shape) if masks is not None else None
                seg_loss = criterion(pred, mask_reconstructed) if mask_reconstructed is not None else 0.0
                running["dice"] += _dice_from_logits_map(pred, mask_reconstructed, threshold=0.5, patch_count=patch_count)
                fpr, fnr, tpr, tnr = _get_pred_rates_from_logits(pred, mask_reconstructed)
                running['px_fpr'] += fpr
                running['px_fnr'] += fnr
                running['px_tpr'] += tpr
                running['px_tnr'] += tnr
            else:
                seg_loss = torch.tensor(0.0, device=device, requires_grad=False)
                n_neg += 1
            cls_loss = torch.nn.functional.cross_entropy(cls_logits, labels.to(device))
            running["seg_loss"] += float(seg_loss.item())
            running["cls_loss"] += float(cls_loss.item())


    cls_metrics = classification_metrics(all_preds, torch.cat(all_labels).numpy())
    if wandb_run is not None:
        wandb_run.summary['test/seg/loss'] = running["seg_loss"] / max(1, n_pos)
        wandb_run.summary['test/cls/loss'] = running["cls_loss"] / max(1, len(dataloader.dataset))
        wandb_run.summary['test/seg/dice'] = running["dice"] / max(1, n_pos)
        wandb_run.summary['test/cls/acc'] = cls_metrics['acc']
        wandb_run.summary['test/cls/auroc'] = cls_metrics['auroc']
        wandb_run.summary['test/cls/auprc'] = cls_metrics['auprc']
        wandb_run.summary['test/seg/px_fpr'] = running['px_fpr'].mean().item() if n_pos > 0 else 0.0
        wandb_run.summary['test/seg/px_fnr'] = running['px_fnr'].mean().item() if n_pos > 0 else 0.0
        wandb_run.summary['test/seg/px_tpr'] = running['px_tpr'].mean().item() if n_pos > 0 else 0.0
        wandb_run.summary['test/seg/px_tnr'] = running['px_tnr'].mean().item() if n_neg > 0 else 0.0


def train(
        model: torch.nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        epochs: int = 10,
        scheduler: Optional[object] = None,
        clip_grad: Optional[float] = None,
        validate_every: int = 1,
        early_stopping_patience: Optional[int] = None,
        save_path: Optional[str] = None,
        min_delta: float = 1e-8,
        wandb_run: Optional[Any] = None,
        ) -> Dict[str, object]:
    """
    High-level training loop.

    Returns history dict with lists for 'train_loss','train_dice','val_loss','val_dice' (when validation available).
    """
    history = {"train_seg_loss": [], "train_cls_loss": [], "train_dice": []}
    if 'val' in dataloaders:
        history.update({"val_seg_loss": [], "val_cls_loss": [], "val_dice": [], "val_detection_rate": []})

    best_val = float('inf')
    best_epoch = -1
    epochs_since_improve = 0
    best_model_path: Optional[str] = None
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_stats = train_epoch(model, dataloaders['train'], optimizer, criterion, device, clip_grad)
        history['train_seg_loss'].append(train_stats['seg_loss'])
        history['train_cls_loss'].append(train_stats['cls_loss'])
        history['train_dice'].append(train_stats['dice'])

        if wandb_run is not None:
            _safe_wandb_log(wandb_run, 'train/seg/loss', train_stats['seg_loss'], step=epoch)
            _safe_wandb_log(wandb_run, 'train/cls/loss', train_stats['cls_loss'], step=epoch)
            _safe_wandb_log(wandb_run, 'train/seg/dice', train_stats['dice'], step=epoch)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        did_validate = False
        if 'val' in dataloaders and ((epoch + 1) % validate_every == 0):
            val_stats = validate(model, dataloaders['val'], criterion, device)
            history['val_seg_loss'].append(val_stats['seg_loss'])
            history['val_cls_loss'].append(val_stats['cls_loss'])
            history['val_dice'].append(val_stats['dice'])

            if wandb_run is not None:
                _safe_wandb_log(wandb_run, 'val/seg/loss', val_stats['seg_loss'], step=epoch)
                _safe_wandb_log(wandb_run, 'val/cls/loss', val_stats['cls_loss'], step=epoch)
                _safe_wandb_log(wandb_run, 'val/seg/dice', val_stats['dice'], step=epoch)
                _safe_wandb_log(wandb_run, 'val/seg/lesion_detected', val_stats['lesion_detected'], step=epoch)
                _safe_wandb_log(wandb_run, 'val/cls/acc', val_stats['cls_metrics']['acc'], step=epoch)
                _safe_wandb_log(wandb_run, 'val/cls/auroc', val_stats['cls_metrics']['auroc'], step=epoch)
                _safe_wandb_log(wandb_run, 'val/cls/auprc', val_stats['cls_metrics']['auprc'], step=epoch)
            did_validate = True

            if early_stopping_patience is not None:
                current_val = val_stats['seg_loss']
                # improvement if decrease greater than min_delta
                if current_val + min_delta < best_val:
                    best_val = current_val
                    best_epoch = epoch
                    epochs_since_improve = 0
                    # save best model
                    if save_path is not None:
                        torch.save(model.state_dict(), save_path)
                        best_model_path = save_path
                        print(f"Saved best model (val loss {best_val:.6f}) to {save_path}")
                        if wandb_run is not None:
                            _safe_wandb_log(wandb_run, 'best/model_path', save_path, step=epoch)
                            _safe_wandb_log(wandb_run, 'best/val_loss', best_val, step=epoch)
                            _safe_wandb_log(wandb_run, 'early_stopping/patience', early_stopping_patience - epochs_since_improve)
                else:
                    epochs_since_improve += 1
                    if wandb_run is not None:
                        _safe_wandb_log(wandb_run, 'early_stopping/patience', early_stopping_patience - epochs_since_improve)

                if epochs_since_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {epochs_since_improve} validation checks.")
                    if wandb_run is not None:
                        _safe_wandb_log(wandb_run, 'early_stopping/stopped_epoch', epoch+1, step=epoch)
                        _safe_wandb_log(wandb_run, 'early_stopping/patience', early_stopping_patience - epochs_since_improve)
                    break

        print(f"Epoch {epoch+1}/{epochs} | TRAIN SL: {train_stats['seg_loss']:.4f} CL: {train_stats['cls_loss']:.4f} D: {train_stats['dice']:.4f}", end='')
        if did_validate:
            print(f" | VAL SL: {val_stats['seg_loss']:.4f} CL: {val_stats['cls_loss']:.4f} D: {val_stats['dice']:.4f}, Detected: {val_stats['lesion_detected']:.4f}")
        else:
            print("")

    history_out: Dict[str, object] = history
    history_out['best_epoch'] = best_epoch
    history_out['best_val_loss'] = best_val if best_epoch >= 0 else None
    history_out['best_model_path'] = best_model_path

    # Final wandb logs
    if wandb_run is not None:
        if best_model_path is not None:
            _safe_wandb_log(wandb_run, 'best/model_path', best_model_path)
            _safe_wandb_log(wandb_run, 'best/val_loss', history_out['best_val_loss'])

    return history_out
