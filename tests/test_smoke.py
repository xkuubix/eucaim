# tests/test_smoke.py
import torch


def get_model():
    from src.models import PatchUNet

    config = {
        "data": {
            "patch_size": 128,
            "overlap_train": 0.0,
            "overlap_eval": 0.0,
            "bag_size": 4,
            "empty_threshold": 0.0,
        },
        "bottleneck_channels": 512,
    }

    return PatchUNet(
        config=config,
        num_classes=2,
        mil_hidden=128,
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
    )


def test_model_instantiates():
    model = get_model()
    assert model is not None
    assert model.mil is not None


def test_forward_seg_only():
    """Segmentation forward pass without mask — 2 iterations."""
    model = get_model().eval()
    sample = torch.randn(1, 256, 256)

    for _ in range(2):
        pred, masks, ids, cls_logits, attn, seg_mask = model(sample)
        assert pred.shape[1] == 1
        assert cls_logits.shape == (1, 2)
        assert attn.shape[1] == 1
        assert masks is None
        assert seg_mask is None


def test_forward_with_mask():
    """Segmentation + mask forward pass — 2 iterations."""
    model = get_model().train()
    sample = torch.randn(1, 256, 256)
    mask = (torch.rand(1, 256, 256) > 0.95).float()

    for _ in range(2):
        pred, masks, ids, cls_logits, attn, seg_mask = model(sample, mask, bg_ratio=1.0)
        assert pred.shape[1] == 1
        assert masks is not None
        assert cls_logits.shape == (1, 2)
        assert seg_mask is not None
        assert seg_mask.dtype == torch.bool
