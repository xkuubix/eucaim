import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Any


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
        dice_per_item = (2 * tp / (2 * tp + fp + fn + 1e-7)).cpu().numpy()
        # if result is scalar (no batch dim), wrap
        if np.isscalar(dice_per_item):
            dice_per_item = np.array([dice_per_item])
    return float(dice_per_item.mean()), int(dice_per_item.size)


def _safe_neptune_log(run: Any, key: str, value: object, step: Optional[int] = None) -> None:
    """Try to log a value to Neptune run. Works whether the run exposes `.log()` or accepts assignment.

    This is intentionally permissive: if logging fails we silently continue so this file stays usable
    even if Neptune isn't available at runtime.
    """
    if run is None:
        return
    try:
        # preferred: run['key'].log(value, step=step)
        target = run[key]
        try:
            if step is None:
                target.log(value)
            else:
                target.log(value, step=step)
            return
        except Exception:
            # fallback to assignment
            run[key] = value
            return
    except Exception:
        # fallback: maybe run supports direct item assignment
        try:
            run[key] = value
        except Exception:
            # give up silently
            return


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
    running_loss = 0.0
    running_dice = 0.0
    n_items = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['annotation'].to(device)

        optimizer.zero_grad()
        preds_patched, masks_patched, _ = model(images, masks)

        loss = criterion(preds_patched, masks_patched)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)

        dice_mean, cnt = _binary_metrics_from_logits(preds_patched, masks_patched)
        running_dice += dice_mean * cnt
        n_items += cnt

    avg_loss = running_loss / max(1, len(dataloader.dataset))
    avg_dice = running_dice / max(1, n_items)
    return {"loss": avg_loss, "dice": avg_dice}


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
    running_loss = 0.0
    running_dice = 0.0
    n_items = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['annotation'].to(device)

            preds_patched, masks_patched, _ = model(images, masks)
            loss = criterion(preds_patched, masks_patched)

            running_loss += float(loss.item()) * images.size(0)
            dice_mean, cnt = _binary_metrics_from_logits(preds_patched, masks_patched)
            running_dice += dice_mean * cnt
            n_items += cnt

    avg_loss = running_loss / max(1, len(dataloader.dataset))
    avg_dice = running_dice / max(1, n_items)
    return {"loss": avg_loss, "dice": avg_dice}


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    return_predictions: bool = False,
    neptune_run: Optional[Any] = None,
) -> Dict[str, object]:
    """
    Run a test loop. If `criterion` is provided, compute loss as well.

    If `return_predictions` is True, returns a list of (instances_ids, probs) per batch.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_items = 0
    all_preds: List[Tuple[object, torch.Tensor]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch.get('annotation')
            if masks is not None:
                masks = masks.to(device)

            preds_patched, masks_patched, instances_ids = model(images, masks if masks is not None else None)

            if criterion is not None and masks is not None:
                loss = criterion(preds_patched, masks_patched)
                running_loss += float(loss.item()) * images.size(0)

            if masks is not None:
                dice_mean, cnt = _binary_metrics_from_logits(preds_patched, masks_patched)
                running_dice += dice_mean * cnt
                n_items += cnt

            if return_predictions:
                probs = torch.sigmoid(preds_patched).cpu()
                all_preds.append((instances_ids, probs))

    out: Dict[str, object] = {}
    if criterion is not None and len(dataloader.dataset) > 0:
        out['loss'] = running_loss / max(1, len(dataloader.dataset))
    if n_items > 0:
        out['dice'] = running_dice / n_items
    if return_predictions:
        out['predictions'] = all_preds
    # Neptune logging if provided
    if neptune_run is not None:
        if 'loss' in out:
            _safe_neptune_log(neptune_run, 'test/loss', out['loss'])
        if 'dice' in out:
            _safe_neptune_log(neptune_run, 'test/dice', out['dice'])
        if return_predictions:
            _safe_neptune_log(neptune_run, 'test/n_predictions', len(all_preds))

    return out


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
    neptune_run: Optional[Any] = None,
 ) -> Dict[str, object]:
    """
    High-level training loop.

    Returns history dict with lists for 'train_loss','train_dice','val_loss','val_dice' (when validation available).
    """
    history = {"train_loss": [], "train_dice": []}
    if 'val' in dataloaders:
        history.update({"val_loss": [], "val_dice": []})

    # Early stopping bookkeeping
    best_val = float('inf')
    best_epoch = -1
    epochs_since_improve = 0
    best_model_path: Optional[str] = None
    if save_path is not None:
        # ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_stats = train_epoch(model, dataloaders['train'], optimizer, criterion, device, clip_grad)
        history['train_loss'].append(train_stats['loss'])
        history['train_dice'].append(train_stats['dice'])

        # Neptune: log training metrics per epoch
        if neptune_run is not None:
            _safe_neptune_log(neptune_run, 'train/loss', train_stats['loss'], step=epoch)
            _safe_neptune_log(neptune_run, 'train/dice', train_stats['dice'], step=epoch)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                # some schedulers expect a metric argument
                pass

        did_validate = False
        if 'val' in dataloaders and ((epoch + 1) % validate_every == 0):
            val_stats = validate(model, dataloaders['val'], criterion, device)
            history['val_loss'].append(val_stats['loss'])
            history['val_dice'].append(val_stats['dice'])
            did_validate = True

            # Early stopping / checkpointing logic (only when we actually validated)
            if early_stopping_patience is not None:
                current_val = val_stats['loss']
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
                        if neptune_run is not None:
                            _safe_neptune_log(neptune_run, 'best/model_path', save_path, step=epoch)
                            _safe_neptune_log(neptune_run, 'best/val_loss', best_val, step=epoch)
                            _safe_neptune_log(neptune_run, 'early_stopping/epochs_since_improve', epochs_since_improve, step=epoch)
                else:
                    epochs_since_improve += 1
                    if neptune_run is not None:
                        _safe_neptune_log(neptune_run, 'early_stopping/epochs_since_improve', epochs_since_improve, step=epoch)

                if epochs_since_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {epochs_since_improve} validation checks.")
                    if neptune_run is not None:
                        _safe_neptune_log(neptune_run, 'early_stopping/stopped_epoch', epoch+1, step=epoch)
                        _safe_neptune_log(neptune_run, 'early_stopping/epochs_since_improve', epochs_since_improve, step=epoch)
                    break

        print(f"Epoch {epoch+1}/{epochs} | train loss: {train_stats['loss']:.4f} dice: {train_stats['dice']:.4f}", end='')
        if did_validate:
            print(f" | val loss: {val_stats['loss']:.4f} val dice: {val_stats['dice']:.4f}")
        else:
            print("")

    # Attach metadata about best model / early stopping
    history_out: Dict[str, object] = history
    history_out['best_epoch'] = best_epoch
    history_out['best_val_loss'] = best_val if best_epoch >= 0 else None
    history_out['best_model_path'] = best_model_path

    # Final neptune logs
    if neptune_run is not None:
        if best_model_path is not None:
            _safe_neptune_log(neptune_run, 'best/model_path', best_model_path)
            _safe_neptune_log(neptune_run, 'best/val_loss', history_out['best_val_loss'])
        _safe_neptune_log(neptune_run, 'training/epochs_ran', epoch+1)
        _safe_neptune_log(neptune_run, 'training/best_epoch', best_epoch)

    return history_out
