import torch
import torch.nn as nn
from monai.networks.nets import UNet
from ImagePatcher import ImagePatcher
import numpy as np
from contextlib import contextmanager


class AttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al. 2018).

    A(h) = softmax( tanh(V h) ⊙ sigmoid(U h) )
    z     = Σ A(h_k) h_k
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier  = nn.Linear(in_dim, num_classes)

    def forward(self, H: torch.Tensor):
        """
        Args:
            H: [N_patches, in_dim]
        Returns:
            logits:  [1, num_classes]
            weights: [N_patches, 1]
        """
        A = self.attention_w(self.attention_V(H) * self.attention_U(H))  # [N, 1]
        A = torch.softmax(A, dim=0)                                       # [N, 1]
        z = (A * H).sum(dim=0, keepdim=True)                              # [1, C]
        return self.classifier(z), A


class PatchUNet(UNet):
    def __init__(self, config, num_classes=None, mil_hidden=128, *args, **kwds):
        super().__init__(*args, **kwds)
        patch_size = config['data'].get('patch_size', 128)
        overlap = config['data'].get('overlap', 0.)
        bag_size = config['data'].get('bag_size', -1)
        empty_thresh = config['data'].get('empty_threshold', 0.)
        self.patcher = ImagePatcher(patch_size=patch_size, overlap=overlap, bag_size=bag_size, empty_thresh=empty_thresh)
        self._bottleneck_features = {}
        self._hooks = []
        self.gap = nn.AdaptiveAvgPool2d(1)

        bottleneck_channels = config.get('bottleneck_channels', 512)
        self.mil = AttentionMIL(bottleneck_channels, mil_hidden, num_classes) \
                   if num_classes is not None else None

    def _register_bottleneck_hook(self, target_channels=512, source_channels=256):
        def hook_fn(module, input, output):
            self._bottleneck_features['bottleneck'] = output

        for name, module in self.model.named_modules():
            if (isinstance(module, nn.Conv2d)
                    and module.out_channels == target_channels
                    and module.in_channels == source_channels):
                parts = name.split('.')
                if len(parts) >= 2:
                    parent = self.model
                    for part in '.'.join(parts[:-2]).split('.'):
                        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
                    self._hooks.append(parent.register_forward_hook(hook_fn))
                    break
        if not self._hooks:
            raise RuntimeError("Bottleneck hook not registered — check channel sizes match the UNet config")

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._bottleneck_features.clear()

    @contextmanager
    def _bottleneck_ctx(self):
        self._register_bottleneck_hook()
        try:
            yield self._bottleneck_features
        finally:
            self._remove_hooks()

    # ------------------------------------------------------------------
    # Patching utilities
    # ------------------------------------------------------------------

    def patch_image(self, x):
        self.patcher.get_tiles(x.shape[1], x.shape[2])
        instances, instances_ids, instances_coords = self.patcher.convert_img_to_bag(x)
        return instances.to(x.device), instances_ids, instances_coords

    def patch_image_and_mask(self, image, mask):
        """Returns ALL patches and masks — no undersampling."""
        mask = mask.squeeze()
        tiles = self.patcher.get_tiles(image.shape[1], image.shape[2])
        instances, instances_idx, instances_coords = self.patcher.convert_img_to_bag(image)

        mask_instances = []
        for patch_coord in instances_coords:
            i_id, j_id = patch_coord
            idx = np.where((tiles[:, 4] == i_id) & (tiles[:, 5] == j_id))[0][0]
            y, x, h, w = tiles[idx, 0:4].astype(int)
            mask_instances.append(mask[y:y+h, x:x+w])
        mask_instances = torch.stack(mask_instances).unsqueeze(1)

        return (instances.to(image.device),
                mask_instances.to(mask.device),
                instances_idx,
                instances_coords)

    def get_seg_loss_mask(self, mask_patches: torch.Tensor,
                          bg_threshold: float = 0.01,
                          bg_ratio: float = 1.0) -> torch.Tensor:
        """Boolean mask selecting which patches to include in seg loss.

        Keeps all positive patches; randomly undersamples negatives.
        At inference, returns all-True (no undersampling).

        Args:
            mask_patches: [N, 1, H, W]
        Returns:
            keep: [N] bool
        """
        roi_frac = mask_patches.flatten(1).mean(dim=1)
        pos_idx  = torch.where(roi_frac >  bg_threshold)[0]
        neg_idx  = torch.where(roi_frac <= bg_threshold)[0]


        if self.training and bg_ratio >= 0:
            if len(pos_idx) == 0:
                min_neg = 32
                # keep a minimum number of negatives
                k = min(neg_idx.numel(), min_neg)  # e.g. 32 or 64
                neg_idx = neg_idx[torch.randperm(len(neg_idx), device=neg_idx.device)[:k]]
            n_neg_keep = int(bg_ratio * len(pos_idx))
            if len(neg_idx) > n_neg_keep:
                neg_idx = neg_idx[torch.randperm(len(neg_idx), device=neg_idx.device)[:n_neg_keep]]

        keep = torch.zeros(mask_patches.shape[0], dtype=torch.bool, device=mask_patches.device)
        keep[pos_idx] = True
        keep[neg_idx] = True
        return keep

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, mask=None, bg_threshold=0.01, bg_ratio=1.0):
        C, H, W = x.shape
        ps = self.patcher.patch_size
        new_H = math.ceil(H / ps) * ps
        new_W = math.ceil(W / ps) * ps

        if new_H != H or new_W != W:
            padded_x = torch.zeros((C, new_H, new_W), device=x.device, dtype=x.dtype)
            padded_x[:, :H, :W] = x
            x = padded_x

            if mask is not None:
                padded_m = torch.zeros((C, new_H, new_W), device=mask.device, dtype=mask.dtype)
                padded_m[:, :H, :W] = mask
                mask = padded_m
        # Full patch set — no undersampling here
        if mask is not None:
            x, mask_patches, instances_ids, _ = self.patch_image_and_mask(x, mask)
        else:
            x, instances_ids, _ = self.patch_image(x)
            mask_patches = None
        # Single forward pass, MIL sees all patches
        with self._bottleneck_ctx() as feats:
            pred_patches = super().forward(x)
            bottleneck = feats.get('bottleneck')

        # Undersampling mask for seg loss (only used during training with mask)
        seg_loss_mask = None
        if mask_patches is not None:
            seg_loss_mask = self.get_seg_loss_mask(mask_patches, bg_threshold, bg_ratio)

        # MIL over all patches
        cls_logits, attn_weights = None, None
        if self.mil is not None and bottleneck is not None:
            H_patches = self.gap(bottleneck).squeeze(-1).squeeze(-1)  # [N, C]
            cls_logits, attn_weights = self.mil(H_patches)

        return pred_patches, mask_patches, instances_ids, cls_logits, attn_weights, seg_loss_mask


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt

    # --- build a minimal config matching default hook channels ---
    config = {
        'data': {'patch_size': 512, 'overlap': 0., 'bag_size': -1, 'empty_threshold': 0.},
        'bottleneck_channels': 512,
    }

    model = PatchUNet(
        config=config,
        num_classes=2,
        mil_hidden=128,
        # UNet kwargs
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
    ).eval()

    sample_input = torch.randn(1, 1024, 1025)
    sample_mask  = (torch.rand(1, 1024, 1025) > 1.95).float()  # sparse positive mask
    image_label  = torch.tensor([1])                          # positive image

    # ------------------------------------------------------------------
    # 1. Segmentation only (no mask → no seg_loss_mask)
    # ------------------------------------------------------------------
    print("=== 1. Seg only ===")
    pred, masks, ids, cls_logits, attn, seg_mask = model(sample_input)
    print(f"  pred shape:      {pred.shape}")          # [N_patches, 1, 128, 128]
    print(f"  cls_logits:      {cls_logits}")          # [1, 2]
    print(f"  attn_weights:    {attn.shape}")          # [N_patches, 1]
    print(f"  seg_loss_mask:   {seg_mask}")            # None

    # ------------------------------------------------------------------
    # 2. Seg + mask → seg_loss_mask computed
    # ------------------------------------------------------------------
    print("\n=== 2. Seg + mask ===")
    model.train()
    pred, masks, ids, cls_logits, attn, seg_mask = model(sample_input, sample_mask, bg_ratio=1.0)
    reconstructed_image = model.patcher.reconstruct_image_from_patches(pred, ids, sample_input.shape)
    print(f"  total patches:   {pred.shape[0]}")
    print(f"  seg mask keeps:  {seg_mask.sum().item()} / {seg_mask.shape[0]} patches")
    print(f"  pred[seg_mask]:  {pred[seg_mask].shape}")
    print(f"  masks[seg_mask]: {masks[seg_mask].shape}")
    print(f"  cls_logits:      {cls_logits}")          # [1, 2]
    print(f"  attn_weights:    {attn.shape}")          # [N_patches, 1]
    print(f"{reconstructed_image[0].shape=}")
    # ------------------------------------------------------------------
    # 3. MIL bag-level prediction
    # ------------------------------------------------------------------
    print("\n=== 3. MIL bag-level ===")
    model.eval()
    pred, masks, ids, cls_logits, attn, seg_mask = model(sample_input, sample_mask)
    print(f"  predicted class: {cls_logits.argmax(dim=-1).item()}")
    print(f"  cls_logits:      {cls_logits}")          # [1, 2]
    print(f"  attn_weights:    min={attn.min():.4f}  max={attn.max():.4f}  sum={attn.sum():.4f}")
    top_k = attn.squeeze(1).topk(3)
    print(f"  top-3 patches:   idx={top_k.indices.tolist()}  scores={[f'{v:.4f}' for v in top_k.values.tolist()]}")

    # ------------------------------------------------------------------
    # 4. Visualisation
    # ------------------------------------------------------------------
    n_show = min(4, pred.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 3, 6))
    for i in range(n_show):
        axes[0, i].imshow(pred[i, 0].detach().cpu(), cmap='gray')
        axes[0, i].set_title(f"pred patch {i}\nattn={attn[i].item():.3f}")
        axes[0, i].axis('off')
        if masks is not None:
            axes[1, i].imshow(masks[i, 0].detach().cpu(), cmap='gray')
            axes[1, i].set_title(f"mask patch {i}")
            axes[1, i].axis('off')
    plt.suptitle(f"Bag pred: class={cls_logits.argmax(dim=-1).item()}  "
                 f"logits={cls_logits.detach().cpu().numpy().round(2)}")
    plt.tight_layout()
    plt.show()