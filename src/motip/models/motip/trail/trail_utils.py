import torch
from torch.nn import functional as F
from torchvision.ops import roi_align


def check_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"NaN/Inf in {name}")


def extract_roi(source, boxes, valid_ratios, roi_size):
    """source [B,C,H,W]; boxes [B,Q,4] cxcywh relative to UNPADDED image.

    The feature canvas includes right/bottom padding. Apply its valid ratio
    before ROIAlign (aligned=True, two samples per bin).
    """
    b, q, _ = boxes.shape
    h, w = source.shape[-2:]
    xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1).clamp(0, 1)
    xyxy = xyxy * valid_ratios[:, None, :].repeat(1, 1, 2)
    xyxy = xyxy * boxes.new_tensor([w, h, w, h])
    batch = torch.arange(b, device=boxes.device)[:, None].expand(b, q)
    rois = torch.cat((batch[..., None], xyxy), dim=-1).reshape(-1, 5)
    # ROIAlign/grid_sample float32 islands also support models under autocast.
    result = roi_align(source.float(), rois.float(), output_size=roi_size,
                       spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return result.reshape(b, q, source.shape[1], *roi_size)


def splat_roi(rois, boxes, height, width):
    """Differentiably place [N,C,Rh,Rw] patches on [N,C,H,W] image canvases.

    Bilinear interpolation in box-local coordinates, with soft box edges to
    reduce complete disappearance of narrow boxes between coarse cell centers.
    Coordinates are global UNPADDED normalized image coordinates.
    """
    n, c = rois.shape[:2]
    if n == 0:
        return rois.new_zeros((0, c, height, width))
    boxes = boxes.float()
    y, x = torch.meshgrid((torch.arange(height, device=boxes.device) + .5) / height,
                          (torch.arange(width, device=boxes.device) + .5) / width,
                          indexing="ij")
    center = boxes[:, None, None, :2]
    size = boxes[:, None, None, 2:].clamp_min(1e-4)
    grid = (torch.stack((x, y), -1)[None] - center) * 2 / size
    patch = F.grid_sample(rois.float(), grid, mode="bilinear", padding_mode="border",
                          align_corners=False)
    # About one-cell soft boundary, with actual zero for degenerate boxes.
    distance = (size / 2 - (torch.stack((x, y), -1)[None] - center).abs())
    occupancy = (distance * boxes.new_tensor([width, height]) * 8).sigmoid().prod(-1)
    valid = (boxes[:, 2:] > 0).all(-1)
    return patch * occupancy[:, None] * valid[:, None, None, None]


def canonical_slots(labels, masks, vocabulary):
    """Compact per-(batch,augmentation group) dictionary of assigned MOTIP IDs.

    Never uses dataset IDs or assumes that the observation's column is its ID.
    Padding IDs are -1; two observations may not update one slot in one frame.
    """
    bg, t, n = labels.shape
    selected = []
    for i in range(bg):
        valid = labels[i][~masks[i]]
        if ((valid < 0) | (valid >= vocabulary)).any():
            raise ValueError("Trail IDs must be assigned MOTIP dictionary slots")
        for j in range(t):
            ids = labels[i, j][~masks[i, j]]
            if ids.unique().numel() != ids.numel():
                raise ValueError("Duplicate active trail slot within a frame")
        selected.append(valid.unique(sorted=True))
    k = max(1, max(x.numel() for x in selected))
    slots = labels.new_full((bg, k), -1)
    for i, ids in enumerate(selected):
        slots[i, :ids.numel()] = ids
    matches = labels[..., None] == slots[:, None, None, :]
    indices = matches.long().argmax(-1)
    return slots, indices
